#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/plugins/base/fake_live_src.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <torch/torch.h>
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

std::filesystem::path fixture_path(const char* name)
{
    return std::filesystem::path(LEAKFLOW_TORCH_FIXTURE_DIR) / name;
}

// Records, per received buffer, its vector-clock slot-1 value (FakeLiveSrc's slot)
// and the row shape -- so the test can assert the per-row index increments.
class CaptureSink final : public leakflow::Element {
public:
    CaptureSink()
        : Element("capture")
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
                last_rows = payload->tensor().size(0);
                last_cols = payload->tensor().dim() >= 2 ? payload->tensor().size(1) : 0;
            }
        }
        return std::nullopt;
    }

    std::atomic<int> count = 0; // atomic so a peer thread can observe progress mid-stream
    std::vector<std::uint32_t> slot1;
    std::int64_t last_rows = 0;
    std::int64_t last_cols = 0;
};

// A one-run (non-live) Torch source, for the liveness-propagation negative case.
class StaticSrc final : public leakflow::Element {
public:
    StaticSrc()
        : Element("static")
    {
        add_output_pad(
            leakflow::Pad("src", leakflow::PadDirection::Output, leakflow::Caps(leakflow::base::torch_tensor_caps_type)));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        auto payload = leakflow::base::TorchTensorPayload(torch::zeros({1, 4}));
        leakflow::Buffer buffer(payload.caps());
        buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
        return buffer;
    }
};

} // namespace

int main()
{
    // Walkthrough Example 4: a single live source streams one Buffer per axis-0
    // row of [50, 5000], run() pumps until EOS, and the provenance index
    // increments per row.
    leakflow::Pipeline pipeline;
    auto src = std::make_shared<leakflow::plugins::base::FakeLiveSrc>("live");
    src->set_property("path", std::string(fixture_path("traces_first_50.pt").string()));
    auto src_handle = pipeline.add(src);
    auto sink = std::make_shared<CaptureSink>();
    auto sink_handle = pipeline.add(sink);
    pipeline.link(src_handle, "src", sink_handle, "in");

    if (!expect(src->is_live(), "FakeLiveSrc did not declare itself live")) {
        return 1;
    }

    (void)pipeline.run();

    if (!expect(sink->count == 50, "FakeLiveSrc did not stream 50 rows")) {
        return 1;
    }
    if (!expect(sink->last_rows == 1 && sink->last_cols == 5000, "streamed row shape was not [1, 5000]")) {
        return 1;
    }

    bool indexes_ok = sink->slot1.size() == 50;
    for (std::size_t i = 0; indexes_ok && i < sink->slot1.size(); ++i) {
        indexes_ok = sink->slot1[i] == static_cast<std::uint32_t>(i + 1);
    }
    if (!expect(indexes_ok, "per-row provenance slot 1 did not increment 1..50")) {
        return 1;
    }

    // Running again replays the stream (FakeLiveSrc::stop() reset the cursor).
    auto sink2 = std::make_shared<CaptureSink>();
    leakflow::Pipeline pipeline2;
    auto src2 = std::make_shared<leakflow::plugins::base::FakeLiveSrc>("live2");
    src2->set_property("path", std::string(fixture_path("plain_texts_first_50.pt").string()));
    auto src2_handle = pipeline2.add(src2);
    auto sink2_handle = pipeline2.add(sink2);
    pipeline2.link(src2_handle, "src", sink2_handle, "in");
    (void)pipeline2.run();
    if (!expect(sink2->count == 50, "FakeLiveSrc did not stream 50 plaintext rows")) {
        return 1;
    }

    // Liveness propagation (S11.5): the live source and everything downstream of
    // it are live-driven; a one-run pipeline has no live-driven elements.
    if (!expect(pipeline.is_live_driven(src_handle), "live source was not live-driven")) {
        return 1;
    }
    if (!expect(pipeline.is_live_driven(sink_handle), "sink downstream of a live source was not live-driven")) {
        return 1;
    }

    leakflow::Pipeline static_pipeline;
    auto static_src = static_pipeline.add(std::make_shared<StaticSrc>());
    auto static_sink = static_pipeline.add(std::make_shared<CaptureSink>());
    static_pipeline.link(static_src, "src", static_sink, "in");
    if (!expect(!static_pipeline.is_live_driven(static_sink), "one-run sink was wrongly live-driven")) {
        return 1;
    }
    if (!expect(!static_pipeline.is_live_driven(static_src), "one-run source was wrongly live-driven")) {
        return 1;
    }

    // Cooperative stop (S11.8), deterministic case: a stop requested before run()
    // makes the live pump exit immediately with zero rows pumped.
    {
        leakflow::Pipeline stop_pipeline;
        auto stop_src = std::make_shared<leakflow::plugins::base::FakeLiveSrc>("live_prestop");
        stop_src->set_property("path", std::string(fixture_path("traces_first_50.pt").string()));
        auto stop_src_handle = stop_pipeline.add(stop_src);
        auto stop_sink = std::make_shared<CaptureSink>();
        auto stop_sink_handle = stop_pipeline.add(stop_sink);
        stop_pipeline.link(stop_src_handle, "src", stop_sink_handle, "in");

        std::stop_source source;
        source.request_stop();
        stop_pipeline.set_stop_token(source.get_token());
        (void)stop_pipeline.run();
        if (!expect(stop_sink->count == 0, "pre-requested stop should pump zero rows")) {
            return 1;
        }
    }

    // Cooperative stop, interruptible-sleep case: a paced source (4 traces/s) is
    // stopped once it has streamed at least one row; the pacing wait wakes promptly
    // so run() returns having pumped only a few of the 50 rows -- never the full
    // stream. Progress-driven (not wall-clock) so a loaded machine cannot flake it;
    // a hang (broken interruptible wait) is caught by the test harness timeout.
    {
        leakflow::Pipeline paced_pipeline;
        auto paced_src = std::make_shared<leakflow::plugins::base::FakeLiveSrc>("live_paced");
        paced_src->set_property("path", std::string(fixture_path("traces_first_50.pt").string()));
        paced_src->set_property("sample_rate_hz", 4.0);
        auto paced_src_handle = paced_pipeline.add(paced_src);
        auto paced_sink = std::make_shared<CaptureSink>();
        auto paced_sink_handle = paced_pipeline.add(paced_sink);
        paced_pipeline.link(paced_src_handle, "src", paced_sink_handle, "in");

        std::stop_source source;
        paced_pipeline.set_stop_token(source.get_token());

        std::thread runner([&paced_pipeline]() { (void)paced_pipeline.run(); });
        // Wait for the first streamed row (generous cap), then request the stop.
        for (int attempt = 0; attempt < 4000 && paced_sink->count.load() < 1; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        source.request_stop();
        runner.join();

        if (!expect(paced_sink->count.load() >= 1, "paced stream did not produce any row before stop")) {
            return 1;
        }
        if (!expect(paced_sink->count.load() < 50, "cooperative stop did not stop the stream early")) {
            return 1;
        }
    }

    return 0;
}
