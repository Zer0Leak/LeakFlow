#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_segments.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
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

leakflow::Caps generic()
{
    return leakflow::Caps(leakflow::generic_buffer_caps_type);
}

// A live source that streams `total` buffers (one per process()), claims a clock
// slot, and reports EOS when drained. stop() resets it for replay.
class LiveSource final : public leakflow::Element {
public:
    LiveSource(std::string name, int total)
        : Element(std::move(name)), total_(total)
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "LiveSource";
        descriptor.klass = "Source/Live";
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.live_source = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        if (cursor_ >= total_) {
            return std::nullopt;
        }
        ++cursor_;
        leakflow::Buffer buffer(generic());
        buffer.set_metadata("row", std::to_string(cursor_));
        return buffer;
    }

    [[nodiscard]] bool at_end_of_stream() const override { return cursor_ >= total_; }
    void stop() override { cursor_ = 0; }

private:
    int total_;
    int cursor_ = 0;
};

// A thread-boundary stand-in for Queue, configured for lossless Block handoff so
// the test can assert every buffer arrives in order across the two threads.
class BlockingQueue final : public leakflow::Element {
public:
    explicit BlockingQueue(std::string name)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "BlockingQueue";
        descriptor.klass = "PassThrough/Queue";
        descriptor.input_pads = {leakflow::Pad("sink", leakflow::PadDirection::Input, generic())};
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.property_specs = {
            leakflow::PropertySpec("max_size", std::int64_t{8}, "queue capacity"),
            leakflow::PropertySpec("drop_oldest", false, "block instead of dropping when full"),
        };
        descriptor.thread_boundary = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }
};

// Records, per received buffer, the clock value at slot 1 (the LiveSource's slot).
class CountingSink final : public leakflow::Element {
public:
    explicit CountingSink(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            count.fetch_add(1);
            const auto& provenance = input->provenance();
            slot1.push_back(provenance.size() > 1 ? provenance[1] : 0u);
        }
        return std::nullopt;
    }

    std::atomic<int> count = 0;
    std::vector<std::uint32_t> slot1;
};

class ThrowingSink final : public leakflow::Element {
public:
    explicit ThrowingSink(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        throw std::invalid_argument("detailed threaded failure");
    }
};

class ErrorObserver final : public leakflow::PipelineObserver {
public:
    void observe(const leakflow::PipelineEvent& event) override
    {
        if (event.kind == leakflow::PipelineEventKind::Error) {
            message = event.message;
        }
    }

    std::string message;
};

// A live source with two outputs carrying the SAME generation (a built-in tee):
// one firing stamps both src0 and src1 with the same own-slot count.
class TwoOutLiveSource final : public leakflow::Element {
public:
    TwoOutLiveSource(std::string name, int total)
        : Element(std::move(name)), total_(total)
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "TwoOutLiveSource";
        descriptor.klass = "Source/Live";
        descriptor.output_pads = {
            leakflow::Pad("src0", leakflow::PadDirection::Output, generic()),
            leakflow::Pad("src1", leakflow::PadDirection::Output, generic()),
        };
        descriptor.live_source = true;
        configure_from_descriptor(descriptor);
    }

    leakflow::ElementOutputs process_pads(leakflow::ElementInputs) override
    {
        leakflow::ElementOutputs outputs;
        if (cursor_ >= total_) {
            return outputs;
        }
        ++cursor_;
        outputs.emplace("src0", leakflow::Buffer(generic()));
        outputs.emplace("src1", leakflow::Buffer(generic()));
        return outputs;
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
    [[nodiscard]] bool at_end_of_stream() const override { return cursor_ >= total_; }
    void stop() override { cursor_ = 0; }

private:
    int total_;
    int cursor_ = 0;
};

// Drops even ancestor generations (slot 1): passes odd, returns nothing for even.
// Used to inject a deterministic gap on one branch of a diamond.
class DropEvenAncestor final : public leakflow::Element {
public:
    explicit DropEvenAncestor(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, generic()));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (!input) {
            return std::nullopt;
        }
        const auto& provenance = input->provenance();
        const auto generation = provenance.size() > 1 ? provenance[1] : 0u;
        if (generation % 2u == 0u) {
            return std::nullopt; // drop even generations
        }
        return input;
    }
};

class Passthrough final : public leakflow::Element {
public:
    explicit Passthrough(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }
};

// A two-input join. Provenance on its output is stamped by the engine (merged
// inputs + own slot), so it just forwards one input's payload.
class Join final : public leakflow::Element {
public:
    explicit Join(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in0", leakflow::PadDirection::Input, generic()));
        add_input_pad(leakflow::Pad("in1", leakflow::PadDirection::Input, generic()));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, generic()));
    }

    leakflow::ElementOutputs process_pads(leakflow::ElementInputs inputs) override
    {
        leakflow::ElementOutputs outputs;
        const auto found = inputs.find("in0");
        leakflow::Buffer buffer =
            (found != inputs.end() && found->second) ? *found->second : leakflow::Buffer(generic());
        outputs.emplace("out", std::move(buffer));
        return outputs;
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
};

// A one-run (non-live) source: emits exactly one buffer, then is done.
class OneRunSource final : public leakflow::Element {
public:
    explicit OneRunSource(std::string name)
        : Element(std::move(name))
    {
        add_output_pad(leakflow::Pad("src", leakflow::PadDirection::Output, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        if (done_) {
            return std::nullopt;
        }
        done_ = true;
        return leakflow::Buffer(generic());
    }
    void stop() override { done_ = false; }

private:
    bool done_ = false;
};

} // namespace

int main()
{
    // A live source behind a Queue: two segments, two threads, with the BufferQueue
    // as the cross-thread handoff. Every buffer must arrive at the sink, in order,
    // carrying the source's incrementing provenance slot (the clock rides through
    // the Queue untouched).
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 50));
        auto queue = pipeline.add(std::make_shared<BlockingQueue>("q"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", queue, "sink");
        pipeline.link(queue, "src", sink, "in");

        if (!expect(decompose_into_segments(pipeline).size() == 2, "live source + queue must be two segments")) {
            return 1;
        }

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        if (!expect(sink_element->count.load() == 50, "all 50 buffers must cross the queue to the sink")) {
            return 1;
        }
        bool ordered = sink_element->slot1.size() == 50;
        for (std::size_t i = 0; ordered && i < sink_element->slot1.size(); ++i) {
            ordered = sink_element->slot1[i] == static_cast<std::uint32_t>(i + 1);
        }
        if (!expect(ordered, "Block handoff must preserve order; provenance slot increments 1..50")) {
            return 1;
        }
    }

    // A Queue-free pipeline is a single segment; run_threaded streams it on one
    // thread, equivalent to run().
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 10));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", sink, "in");

        if (!expect(decompose_into_segments(pipeline).size() == 1, "queue-free pipeline is one segment")) {
            return 1;
        }

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());
        if (!expect(sink_element->count.load() == 10, "single-segment threaded run must stream all 10")) {
            return 1;
        }
    }

    // Threaded failures preserve the originating exception text in the observer
    // event used by --graph instead of collapsing it to a generic message.
    {
        leakflow::Pipeline pipeline;
        auto observer = std::make_shared<ErrorObserver>();
        pipeline.set_observer(observer);
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 1));
        auto queue = pipeline.add(std::make_shared<BlockingQueue>("q"));
        auto sink = pipeline.add(std::make_shared<ThrowingSink>("throwing"));
        pipeline.link(src, "src", queue, "sink");
        pipeline.link(queue, "src", sink, "in");

        bool threw = false;
        std::string message;
        try {
            std::stop_source source;
            (void)pipeline.run_threaded(source.get_token());
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        if (!expect(threw && message == "detailed threaded failure",
                "threaded runner did not rethrow the originating exception")) {
            return 1;
        }
        if (!expect(observer->message == "detailed threaded failure",
                "threaded runner did not expose the originating error to observers")) {
            return 1;
        }
    }

    // Cooperative stop, deterministic: a stop requested before run_threaded makes
    // every segment exit immediately, so no buffer reaches the sink.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 50));
        auto queue = pipeline.add(std::make_shared<BlockingQueue>("q"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", queue, "sink");
        pipeline.link(queue, "src", sink, "in");

        std::stop_source source;
        source.request_stop();
        (void)pipeline.run_threaded(source.get_token());

        if (!expect(sink_element->count.load() == 0, "pre-requested stop must pump nothing")) {
            return 1;
        }
    }

    // Aggregator, Barrier realign across a drop. A diamond: source ⟶ {drop-even
    // branch}+{passthrough branch} ⟶ two queues ⟶ join. One branch deterministically
    // drops even generations, so the join must realign by vector clock and fire only
    // on the generations present on BOTH branches: 1, 3, 5, 7, 9.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<TwoOutLiveSource>("src", 10));
        auto odd = pipeline.add(std::make_shared<DropEvenAncestor>("odd"));
        auto all = pipeline.add(std::make_shared<Passthrough>("all"));
        auto q_odd = pipeline.add(std::make_shared<BlockingQueue>("q_odd"));
        auto q_all = pipeline.add(std::make_shared<BlockingQueue>("q_all"));
        auto join = pipeline.add(std::make_shared<Join>("join"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src0", odd, "in");
        pipeline.link(src, "src1", all, "in");
        pipeline.link(odd, "out", q_odd, "sink");
        pipeline.link(all, "out", q_all, "sink");
        pipeline.link(q_odd, "src", join, "in0");
        pipeline.link(q_all, "src", join, "in1");
        pipeline.link(join, "out", sink, "in");

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        const std::vector<std::uint32_t> expected{1u, 3u, 5u, 7u, 9u};
        if (!expect(sink_element->slot1 == expected,
                    "Barrier realign must fire only on generations present on both branches (1,3,5,7,9)")) {
            return 1;
        }
    }

    // Aggregator, Held reuse. A live trace stream + a one-run "key". The key is
    // not live-driven, so it is Held: pulled once and reused for every trace, and
    // its EOS does not end the join. The join fires once per trace (all 8).
    {
        leakflow::Pipeline pipeline;
        auto traces = pipeline.add(std::make_shared<LiveSource>("traces", 8));
        auto key = pipeline.add(std::make_shared<OneRunSource>("key"));
        auto q_traces = pipeline.add(std::make_shared<BlockingQueue>("q_traces"));
        auto q_key = pipeline.add(std::make_shared<BlockingQueue>("q_key"));
        auto join = pipeline.add(std::make_shared<Join>("join"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(traces, "src", q_traces, "sink");
        pipeline.link(key, "src", q_key, "sink");
        pipeline.link(q_traces, "src", join, "in0");
        pipeline.link(q_key, "src", join, "in1");
        pipeline.link(join, "out", sink, "in");

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        // The traces source is slot 1; the join fires once per trace, 1..8.
        const std::vector<std::uint32_t> expected{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
        if (!expect(sink_element->slot1 == expected,
                    "Held key must be reused for every trace (join fires once per trace, 1..8)")) {
            return 1;
        }
    }

    return 0;
}
