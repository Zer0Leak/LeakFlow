#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/progress_sink.hpp"
#include "leakflow/plugins/base/app_src.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

// Records, per received buffer, the vector-clock value on AppSrc's slot (slot 1)
// and a payload marker, so the test can assert both pads of a frame share a clock.
class CaptureSink final : public leakflow::Element {
public:
    explicit CaptureSink(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(
            leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps(leakflow::base::torch_tensor_caps_type)));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            ++count;
            const auto& provenance = input->provenance();
            slot1.push_back(provenance.size() > 1 ? provenance[1] : 0u);
            if (const auto payload = input->payload_as<leakflow::base::TorchTensorPayload>()) {
                markers.push_back(payload->tensor().item<std::int64_t>());
            }
        }
        return std::nullopt;
    }

    int count = 0;
    std::vector<std::uint32_t> slot1;
    std::vector<std::int64_t> markers;
};

[[nodiscard]] leakflow::Buffer marker_buffer(std::int64_t marker)
{
    auto payload = std::make_shared<leakflow::base::TorchTensorPayload>(torch::tensor(marker));
    leakflow::Buffer buffer(payload->caps());
    buffer.set_payload(payload);
    return buffer;
}

int run_tests()
{
    using leakflow::plugins::base::AppSrc;

    // process() (the single-path entry) is invalid for a multi-pad source.
    {
        AppSrc lone("lone");
        bool threw = false;
        try {
            (void)lone.process(std::nullopt);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!expect(threw, "AppSrc::process should reject the single-buffer path")) {
            return 1;
        }
    }

    leakflow::Pipeline pipeline;
    auto source = pipeline.add(std::make_shared<AppSrc>("src"));
    auto sink_a = std::make_shared<CaptureSink>("sink_a");
    auto sink_b = std::make_shared<CaptureSink>("sink_b");
    pipeline.add(sink_a);
    pipeline.add(sink_b);

    pipeline.link(source, "src_0", sink_a, "in");
    pipeline.link(source, "src_1", sink_b, "in");

    auto* app_src = dynamic_cast<AppSrc*>(source.get());
    if (!expect(app_src != nullptr, "added element should be an AppSrc")) {
        return 1;
    }

    if (!expect(!app_src->at_end_of_stream(), "AppSrc should not be at EOS before end_of_stream")) {
        return 1;
    }

    // Two frames, each an aligned pair routed to src_0/src_1.
    app_src->push_frame({marker_buffer(10), marker_buffer(11)});
    app_src->push_frame({marker_buffer(20), marker_buffer(21)});
    app_src->end_of_stream();

    // push after end_of_stream is rejected.
    {
        bool threw = false;
        try {
            app_src->push_frame({marker_buffer(99)});
        } catch (const std::logic_error&) {
            threw = true;
        }
        if (!expect(threw, "push_frame after end_of_stream should throw")) {
            return 1;
        }
    }

    (void)pipeline.run();

    if (!expect(app_src->at_end_of_stream(), "AppSrc should be at EOS after the queue drains")) {
        return 1;
    }
    if (!expect(sink_a->count == 2 && sink_b->count == 2, "each sink should receive one buffer per frame")) {
        return 1;
    }
    if (!expect(sink_a->markers == std::vector<std::int64_t>{10, 20}, "src_0 markers were wrong")) {
        return 1;
    }
    if (!expect(sink_b->markers == std::vector<std::int64_t>{11, 21}, "src_1 markers were wrong")) {
        return 1;
    }

    // Shared provenance: both pads of a frame carry the same AppSrc-slot clock, and
    // it increments across frames. This is what lets a downstream join pair the
    // per-frame outputs (no Sync element needed).
    if (!expect(sink_a->slot1.size() == 2 && sink_b->slot1.size() == 2, "expected two provenance samples per sink")) {
        return 1;
    }
    if (!expect(sink_a->slot1[0] == sink_b->slot1[0] && sink_a->slot1[1] == sink_b->slot1[1],
                "both pads of a frame should share the AppSrc clock value")) {
        return 1;
    }
    if (!expect(sink_a->slot1[0] != sink_a->slot1[1], "the AppSrc clock should increment across frames")) {
        return 1;
    }

    // Pull mode + restart: the producer is index-based and AppSrc rewinds on start(),
    // so a Stop -> Start cycle re-streams from frame 0 (regression: an exhausted
    // source emits nothing and a fresh sweep fails "missing upstream output").
    {
        AppSrc pull("pull");
        pull.set_frame_producer(
            [](std::size_t index, const auto&) -> std::optional<std::vector<leakflow::Buffer>> {
                if (index >= 2) {
                    return std::nullopt;
                }
                std::vector<leakflow::Buffer> frame;
                frame.push_back(marker_buffer(static_cast<std::int64_t>(index)));
                return frame;
            });

        const auto drain = [&pull]() {
            std::vector<std::int64_t> seen;
            pull.start(); // Stopped -> Running rewind
            while (!pull.at_end_of_stream()) {
                auto outputs = pull.process_pads({});
                if (outputs.empty()) {
                    break;
                }
                seen.push_back(
                    outputs.at("src_0").payload_as<leakflow::base::TorchTensorPayload>()->tensor().item<std::int64_t>());
            }
            return seen;
        };

        const auto first = drain();
        const auto second = drain(); // restart
        if (!expect(first == std::vector<std::int64_t>{0, 1}, "pull mode should stream frames 0,1")) {
            return 1;
        }
        if (!expect(second == std::vector<std::int64_t>{0, 1}, "restart should re-stream frames 0,1 after start()")) {
            return 1;
        }
    }

    // Application-driven progress: the reporter handed to the pull producer reaches
    // this element's progress channel, so an application can drive the source's
    // --graph bar without touching the protected API. The framework owns the
    // plumbing; the app supplies only the fraction/message it alone knows.
    {
        struct CapturingProgressSink final : leakflow::ProgressSink {
            void report(leakflow::Element&, const leakflow::ElementProgress& progress) override
            {
                reports.push_back(progress);
            }
            std::vector<leakflow::ElementProgress> reports;
        } progress_sink;

        AppSrc progress_src("progress_src");
        progress_src.set_progress_sink(&progress_sink);
        progress_src.set_frame_producer(
            [](std::size_t index, const auto& report) -> std::optional<std::vector<leakflow::Buffer>> {
                if (index >= 3) {
                    report(1.0, "done", 3, 3);
                    return std::nullopt;
                }
                report(static_cast<double>(index) / 3.0, "frame", index, 3);
                std::vector<leakflow::Buffer> frame;
                frame.push_back(marker_buffer(static_cast<std::int64_t>(index)));
                return frame;
            });

        progress_src.start();
        while (!progress_src.at_end_of_stream()) {
            if (progress_src.process_pads({}).empty()) {
                break;
            }
        }

        // The first report (frame-0 prefetch in start()) and the terminal 1.0 always
        // flush; intermediate ticks may be coalesced by the ~30 Hz throttle. Reaching
        // the sink at all proves the app-driven channel is wired; the terminal report
        // proves fraction 1.0 is promoted to Completed.
        if (!expect(!progress_sink.reports.empty()
                    && progress_sink.reports.front().fraction == 0.0
                    && progress_sink.reports.back().fraction == 1.0
                    && progress_sink.reports.back().status == leakflow::ProgressStatus::Completed,
                    "AppSrc pull producer progress should reach the sink and complete")) {
            return 1;
        }
    }

    return 0;
}

} // namespace

int main()
{
    return run_tests();
}
