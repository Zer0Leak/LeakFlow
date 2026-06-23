// Capstone live-phase test (design doc S11.4 / remaining-work list): a real Sync
// element pairs two independent fake live streams, and a downstream default-Barrier
// join consumes them -- all through the threaded runner. This proves the
// independent-source story end to end: Sync injects a common ancestor so two
// otherwise-unrelated streams become matchable, and the aggregator's default
// Barrier then pairs them for free.

#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/plugins/core/queue.hpp"
#include "leakflow/plugins/core/sync.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
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

leakflow::Caps any_caps()
{
    return leakflow::Caps(leakflow::any_caps_type);
}

// An independent fake live stream: emits `total` buffers, one per process(), claims
// its own clock slot, and reports EOS when drained.
class LiveSource final : public leakflow::Element {
public:
    LiveSource(std::string name, int total)
        : Element(std::move(name)), total_(total)
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "LiveSource";
        descriptor.klass = "Source/Live";
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, any_caps())};
        descriptor.live_source = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        if (cursor_ >= total_) {
            return std::nullopt;
        }
        ++cursor_;
        return leakflow::Buffer(any_caps());
    }

    [[nodiscard]] bool at_end_of_stream() const override { return cursor_ >= total_; }
    void stop() override { cursor_ = 0; }

private:
    int total_;
    int cursor_ = 0;
};

// A two-input default-Barrier join (no policy: it relies on the engine's per-slot
// fold-match). Forwards in0's payload; the engine stamps the merged provenance.
class Join final : public leakflow::Element {
public:
    explicit Join(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in0", leakflow::PadDirection::Input, any_caps()));
        add_input_pad(leakflow::Pad("in1", leakflow::PadDirection::Input, any_caps()));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, any_caps()));
    }

    leakflow::ElementOutputs process_pads(leakflow::ElementInputs inputs) override
    {
        leakflow::ElementOutputs outputs;
        const auto found = inputs.find("in0");
        leakflow::Buffer buffer =
            (found != inputs.end() && found->second) ? *found->second : leakflow::Buffer(any_caps());
        outputs.emplace("out", std::move(buffer));
        return outputs;
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
};

class RecordingSink final : public leakflow::Element {
public:
    explicit RecordingSink(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, any_caps()));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            clocks.push_back(input->provenance());
        }
        return std::nullopt;
    }

    std::vector<std::vector<std::uint32_t>> clocks;
};

std::uint32_t at(const std::vector<std::uint32_t>& clock, std::uint32_t index)
{
    return index < clock.size() ? clock[index] : 0u;
}

} // namespace

int main()
{
    using leakflow::plugins::core::Queue;
    using leakflow::plugins::core::Sync;

    leakflow::Pipeline pipeline;
    auto a = pipeline.add(std::make_shared<LiveSource>("a", 10));
    auto qa = pipeline.add(std::make_shared<Queue>("qa"));
    auto b = pipeline.add(std::make_shared<LiveSource>("b", 10));
    auto qb = pipeline.add(std::make_shared<Queue>("qb"));
    auto sync = pipeline.add(std::make_shared<Sync>("sync"));
    auto join = pipeline.add(std::make_shared<Join>("join"));
    auto sink_element = std::make_shared<RecordingSink>("sink");
    auto sink = pipeline.add(sink_element);

    // Lossless Block queues so the two streams stay in lockstep (no drops).
    qa->set_property("max_size", std::int64_t{8});
    qa->set_property("drop_oldest", false);
    qb->set_property("max_size", std::int64_t{8});
    qb->set_property("drop_oldest", false);

    pipeline.link(a, "src", qa, "sink");
    pipeline.link(b, "src", qb, "sink");
    pipeline.link(qa, "src", sync, "in_0");
    pipeline.link(qb, "src", sync, "in_1");
    pipeline.link(sync, "out_0", join, "in0");
    pipeline.link(sync, "out_1", join, "in1");
    pipeline.link(join, "out", sink, "in");

    // The graph threads: two live sources behind Queues -> 3 segments
    // ({a},{b},{sync,join,sink}).
    if (!expect(pipeline.should_run_threaded(), "two live sources + queues must run threaded")) {
        return 1;
    }

    // Resolve the clock slot each element claims, so the assertions don't depend on
    // add order.
    std::map<std::string, std::uint32_t> base;
    for (const auto& element : pipeline.topology_snapshot().elements) {
        if (element.provenance_slots > 0) {
            base[element.name] = element.provenance_base;
        }
    }

    std::stop_source source;
    (void)pipeline.run_threaded(source.get_token());

    // The downstream Barrier join fires once per aligned pair: all 10.
    if (!expect(sink_element->clocks.size() == 10, "Sync + Barrier must fire once per aligned pair (10)")) {
        return 1;
    }

    const auto a_slot = base.at("a");
    const auto b_slot = base.at("b");
    const auto sync_slot = base.at("sync");

    bool ok = true;
    for (std::size_t i = 0; i < sink_element->clocks.size(); ++i) {
        const auto& clock = sink_element->clocks[i];
        const auto generation = static_cast<std::uint32_t>(i + 1);
        // Common ancestor injected by Sync increments per firing...
        ok = ok && at(clock, sync_slot) == generation;
        // ...and stream A's n-th buffer is paired with stream B's n-th buffer.
        ok = ok && at(clock, a_slot) == generation;
        ok = ok && at(clock, b_slot) == generation;
    }
    if (!expect(ok, "each fire must pair A[n] with B[n] under one shared Sync ancestor n")) {
        return 1;
    }

    return 0;
}
