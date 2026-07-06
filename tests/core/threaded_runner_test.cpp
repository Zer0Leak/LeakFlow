#include "leakflow/core/active_element.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/core/buffer_queue.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_segments.hpp"
#include "leakflow/core/thread_boundary_runtime.hpp"

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <thread>
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

// A source that owns its producer task and pushes through RuntimeContext instead
// of being pumped by the segment runner.
class ActiveLiveSource final : public leakflow::Element, public leakflow::ActiveElement {
public:
    ActiveLiveSource(std::string name, int total)
        : Element(std::move(name)), total_(total)
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "ActiveLiveSource";
        descriptor.klass = "Source/Live";
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.live_source = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        return std::nullopt;
    }

    void start_active(leakflow::RuntimeContext &context) override
    {
        emitted_.store(0);
        worker_ = std::jthread([this, &context](std::stop_token local_stop) {
            try {
                for (int index = 1; index <= total_; ++index) {
                    context.safe_point(*this);
                    if (local_stop.stop_requested() || context.stop_requested()) {
                        break;
                    }
                    leakflow::Buffer buffer(generic());
                    buffer.set_metadata("row", std::to_string(index));
                    if (!context.push(*this, "src", std::move(buffer))) {
                        break;
                    }
                    emitted_.store(index);
                }
                context.end_of_stream(*this, "src");
            } catch (const std::exception &error) {
                context.report_error(*this, error.what());
            } catch (...) {
                context.report_error(*this, "unknown active source failure");
            }
        });
    }

    void stop_active() noexcept override
    {
        worker_ = std::jthread{};
    }

    void wait_active() override
    {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    [[nodiscard]] bool at_end_of_stream() const override { return emitted_.load() >= total_; }
    void stop() override { emitted_.store(0); }

private:
    int total_;
    std::atomic<int> emitted_ = 0;
    std::jthread worker_;
};

// A thread-boundary stand-in for Queue, configured for lossless Block handoff so
// the test can assert every buffer arrives in order across the two threads.
class BlockingQueue final : public leakflow::Element, public leakflow::ThreadBoundaryRuntime {
public:
    explicit BlockingQueue(std::string name)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "BlockingQueue";
        descriptor.klass = "PassThrough/Flow/Queue";
        descriptor.input_pads = {leakflow::Pad("sink", leakflow::PadDirection::Input, generic())};
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.property_specs = {
            leakflow::PropertySpec("max_size", std::int64_t{8}, "queue capacity"),
            leakflow::PropertySpec("drop_oldest", false, "block instead of dropping when full"),
        };
        descriptor.provenance_slots = 0;
        descriptor.thread_boundary = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }

    void prepare_thread_boundary_runtime(std::mutex*) override
    {
        const auto max_size = static_cast<std::size_t>(
            std::max<std::int64_t>(1, property_as<std::int64_t>("max_size").value_or(8)));
        const bool drop_oldest = property_as<bool>("drop_oldest").value_or(true);
        runtime_queue_ = std::make_unique<leakflow::BufferQueue>(
            max_size, drop_oldest ? leakflow::QueueDropPolicy::DropOldest : leakflow::QueueDropPolicy::Block);
    }

    void clear_thread_boundary_runtime() noexcept override
    {
        if (runtime_queue_) {
            runtime_queue_->close();
            runtime_queue_.reset();
        }
    }

    [[nodiscard]] bool boundary_push(leakflow::Buffer buffer, std::stop_token stop) override
    {
        return runtime_queue().push(std::move(buffer), stop);
    }

    [[nodiscard]] leakflow::BufferQueue::Pull boundary_pull(std::stop_token stop) override
    {
        return runtime_queue().pull(stop);
    }

    [[nodiscard]] leakflow::BufferQueue::Pull boundary_try_pull() override
    {
        return runtime_queue().try_pull();
    }

    void boundary_close() override { runtime_queue().close(); }

private:
    [[nodiscard]] leakflow::BufferQueue &runtime_queue()
    {
        if (runtime_queue_ == nullptr) {
            throw std::logic_error("BlockingQueue runtime is not prepared");
        }
        return *runtime_queue_;
    }

    std::unique_ptr<leakflow::BufferQueue> runtime_queue_;
};

// Same boundary behavior as BlockingQueue, but also owns a source-side task: once
// the producer segment enqueues buffers, this element pulls and pushes downstream
// through RuntimeContext. This exercises the passive push collector.
class ActiveBlockingQueue final
    : public leakflow::Element
    , public leakflow::ThreadBoundaryRuntime
    , public leakflow::ActiveElement {
public:
    explicit ActiveBlockingQueue(std::string name)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "ActiveBlockingQueue";
        descriptor.klass = "PassThrough/Flow/Queue";
        descriptor.input_pads = {leakflow::Pad("sink", leakflow::PadDirection::Input, generic())};
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.property_specs = {
            leakflow::PropertySpec("max_size", std::int64_t{8}, "queue capacity"),
            leakflow::PropertySpec("drop_oldest", false, "block instead of dropping when full"),
        };
        descriptor.provenance_slots = 0;
        descriptor.thread_boundary = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }

    void prepare_thread_boundary_runtime(std::mutex*) override
    {
        const auto max_size = static_cast<std::size_t>(
            std::max<std::int64_t>(1, property_as<std::int64_t>("max_size").value_or(8)));
        const bool drop_oldest = property_as<bool>("drop_oldest").value_or(true);
        runtime_queue_ = std::make_unique<leakflow::BufferQueue>(
            max_size, drop_oldest ? leakflow::QueueDropPolicy::DropOldest : leakflow::QueueDropPolicy::Block);
    }

    void clear_thread_boundary_runtime() noexcept override
    {
        if (runtime_queue_) {
            runtime_queue_->close();
        }
        if (worker_.joinable()) {
            worker_ = std::jthread{};
        }
        runtime_queue_.reset();
    }

    [[nodiscard]] bool boundary_push(leakflow::Buffer buffer, std::stop_token stop) override
    {
        return runtime_queue().push(std::move(buffer), stop);
    }

    [[nodiscard]] leakflow::BufferQueue::Pull boundary_pull(std::stop_token stop) override
    {
        return runtime_queue().pull(stop);
    }

    [[nodiscard]] leakflow::BufferQueue::Pull boundary_try_pull() override
    {
        return runtime_queue().try_pull();
    }

    void boundary_close() override { runtime_queue().close(); }

    void start_active(leakflow::RuntimeContext &context) override
    {
        if (worker_.joinable()) {
            stop_active();
        }
        worker_ = std::jthread([this, &context](std::stop_token local_stop) {
            try {
                while (!local_stop.stop_requested() && !context.stop_requested()) {
                    context.safe_point(*this);
                    if (local_stop.stop_requested() || context.stop_requested()) {
                        break;
                    }
                    auto pull = boundary_pull(context.stop_token());
                    if (pull.buffer) {
                        if (!context.push(*this, "src", std::move(*pull.buffer))) {
                            break;
                        }
                        continue;
                    }
                    if (pull.end_of_stream) {
                        context.end_of_stream(*this, "src");
                        break;
                    }
                    break;
                }
            } catch (const std::exception &error) {
                context.report_error(*this, error.what());
            } catch (...) {
                context.report_error(*this, "unknown active queue failure");
            }
        });
    }

    void stop_active() noexcept override
    {
        if (runtime_queue_) {
            runtime_queue_->close();
        }
        if (worker_.joinable()) {
            worker_ = std::jthread{};
        }
    }

    void wait_active() override
    {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    [[nodiscard]] leakflow::BufferQueue &runtime_queue()
    {
        if (runtime_queue_ == nullptr) {
            throw std::logic_error("ActiveBlockingQueue runtime is not prepared");
        }
        return *runtime_queue_;
    }

    std::unique_ptr<leakflow::BufferQueue> runtime_queue_;
    std::jthread worker_;
};

class DescriptorOnlyBoundary final : public leakflow::Element {
public:
    explicit DescriptorOnlyBoundary(std::string name)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "DescriptorOnlyBoundary";
        descriptor.klass = "PassThrough/Flow/Queue";
        descriptor.input_pads = {leakflow::Pad("sink", leakflow::PadDirection::Input, generic())};
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
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

// Records duration (profiling) telemetry events for the live --graph timing overlay.
class DurationTelemetryObserver final : public leakflow::PipelineObserver {
public:
    void observe(const leakflow::PipelineEvent& event) override
    {
        if (event.kind != leakflow::PipelineEventKind::TelemetryChanged || !event.telemetry_change) {
            return;
        }
        if (event.telemetry_change->unit != "ms") {
            return;
        }
        const std::lock_guard<std::mutex> lock(mutex_);
        durations.push_back(event.telemetry_change->element.element_name + "." + event.telemetry_change->telemetry_name);
    }

    std::mutex mutex_;
    std::vector<std::string> durations;
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

// A replayable passive source, like a file source in a live graph: every process()
// call emits the same static value. Used to ensure a consumer segment that has
// active Queue inputs still runs normally when it also contains an internal source.
class StaticSource final : public leakflow::Element {
public:
    explicit StaticSource(std::string name)
        : Element(std::move(name))
    {
        add_output_pad(leakflow::Pad("src", leakflow::PadDirection::Output, generic()));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        ++count;
        return leakflow::Buffer(generic());
    }

    std::atomic<int> count = 0;
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

    // Live --graph timing overlay: with profiling on (and size telemetry off), the
    // runtime publisher must surface per-element process timing as Duration (ms)
    // telemetry for non-boundary elements, so the graph panel can render it.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 30));
        auto queue = pipeline.add(std::make_shared<BlockingQueue>("q"));
        auto sink = pipeline.add(std::make_shared<CountingSink>("sink"));
        pipeline.link(src, "src", queue, "sink");
        pipeline.link(queue, "src", sink, "in");

        auto observer = std::make_shared<DurationTelemetryObserver>();
        pipeline.set_observer(observer);
        pipeline.set_profiling_enabled(true);

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token(), {}, nullptr, /*telemetry_enabled=*/false);

        const std::lock_guard<std::mutex> lock(observer->mutex_);
        const bool saw_sink_process = std::any_of(observer->durations.begin(), observer->durations.end(),
            [](const std::string& entry) { return entry == "sink.process"; });
        if (!expect(saw_sink_process,
                    "profiling overlay must publish per-element process timing (Duration ms) with size telemetry off")) {
            return 1;
        }
    }

    // A boundary element may own its source-side task. The producing segment still
    // pushes into the boundary, but the boundary task pulls and uses RuntimeContext
    // to push directly into the passive downstream sink.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 12));
        auto queue = pipeline.add(std::make_shared<ActiveBlockingQueue>("q"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", queue, "sink");
        pipeline.link(queue, "src", sink, "in");

        if (!expect(decompose_into_segments(pipeline).size() == 2,
                    "live source + active boundary queue must still be two segments")) {
            return 1;
        }

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        if (!expect(sink_element->count.load() == 12,
                    "active boundary queue must push all 12 buffers into the passive sink")) {
            return 1;
        }
        bool ordered = sink_element->slot1.size() == 12;
        for (std::size_t i = 0; ordered && i < sink_element->slot1.size(); ++i) {
            ordered = sink_element->slot1[i] == static_cast<std::uint32_t>(i + 1);
        }
        if (!expect(ordered, "active boundary push must preserve upstream provenance 1..12")) {
            return 1;
        }
    }

    // An ActiveElement source owns its producer task and pushes buffers through
    // RuntimeContext into the Queue boundary; the downstream segment still pulls
    // and processes normally.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<ActiveLiveSource>("active_src", 25));
        auto queue = pipeline.add(std::make_shared<BlockingQueue>("q"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", queue, "sink");
        pipeline.link(queue, "src", sink, "in");

        if (!expect(decompose_into_segments(pipeline).size() == 2,
                    "active source + queue must still decompose into two segments")) {
            return 1;
        }

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        if (!expect(sink_element->count.load() == 25,
                    "all 25 active-source buffers must cross the queue to the sink")) {
            return 1;
        }
        bool ordered = sink_element->slot1.size() == 25;
        for (std::size_t i = 0; ordered && i < sink_element->slot1.size(); ++i) {
            ordered = sink_element->slot1[i] == static_cast<std::uint32_t>(i + 1);
        }
        if (!expect(ordered, "RuntimeContext push must stamp active-source provenance 1..25")) {
            return 1;
        }
    }

    // Active source -> active boundary -> passive sink is an all-active linear
    // handoff: the runner must wait for both active tasks to finish naturally
    // before teardown. This used to require falling back to the pull-side consumer.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<ActiveLiveSource>("active_src", 25));
        auto queue = pipeline.add(std::make_shared<ActiveBlockingQueue>("q"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", queue, "sink");
        pipeline.link(queue, "src", sink, "in");

        if (!expect(decompose_into_segments(pipeline).size() == 2,
                    "active source + active boundary queue must still decompose into two segments")) {
            return 1;
        }

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        if (!expect(sink_element->count.load() == 25,
                    "all-active linear runtime must deliver all 25 buffers to the passive sink")) {
            return 1;
        }
        bool ordered = sink_element->slot1.size() == 25;
        for (std::size_t i = 0; ordered && i < sink_element->slot1.size(); ++i) {
            ordered = sink_element->slot1[i] == static_cast<std::uint32_t>(i + 1);
        }
        if (!expect(ordered, "all-active linear runtime must preserve active-source provenance 1..25")) {
            return 1;
        }
    }

    // Active Queue -> passive transform -> active Queue. This is the composed
    // boundary case: the push collector must route through the passive chain into
    // the next Queue without deadlocking on queue backpressure.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 16));
        auto q1 = pipeline.add(std::make_shared<ActiveBlockingQueue>("q1"));
        auto pass = pipeline.add(std::make_shared<Passthrough>("pass"));
        auto q2 = pipeline.add(std::make_shared<ActiveBlockingQueue>("q2"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", q1, "sink");
        pipeline.link(q1, "src", pass, "in");
        pipeline.link(pass, "out", q2, "sink");
        pipeline.link(q2, "src", sink, "in");

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        if (!expect(sink_element->count.load() == 16,
                    "active Queue -> passive -> active Queue must deliver all 16 buffers")) {
            return 1;
        }
        bool ordered = sink_element->slot1.size() == 16;
        for (std::size_t i = 0; ordered && i < sink_element->slot1.size(); ++i) {
            ordered = sink_element->slot1[i] == static_cast<std::uint32_t>(i + 1);
        }
        if (!expect(ordered, "active Queue -> passive -> active Queue must preserve provenance 1..16")) {
            return 1;
        }
    }

    // Active-capable Queue input + an internal passive source in the same
    // consumer segment. The segment must NOT be skipped as "fully active-driven";
    // otherwise the static source never runs and the join never fires. This is the
    // reduced shape of live traces/plaintexts feeding CPA while TorchFileSrc feeds
    // AttackStats.truth directly.
    {
        leakflow::Pipeline pipeline;
        auto traces = pipeline.add(std::make_shared<LiveSource>("traces", 5));
        auto q_traces = pipeline.add(std::make_shared<ActiveBlockingQueue>("q_traces"));
        auto key_element = std::make_shared<StaticSource>("key");
        auto key = pipeline.add(key_element);
        auto join = pipeline.add(std::make_shared<Join>("join"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(traces, "src", q_traces, "sink");
        pipeline.link(q_traces, "src", join, "in0");
        pipeline.link(key, "src", join, "in1");
        pipeline.link(join, "out", sink, "in");

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        const std::vector<std::uint32_t> expected{1u, 2u, 3u, 4u, 5u};
        if (!expect(sink_element->slot1 == expected,
                    "consumer segment with active Queue input and internal source must fire for every trace")) {
            return 1;
        }
        if (!expect(key_element->count.load() == 5,
                    "internal static source must be executed by the consumer segment thread")) {
            return 1;
        }
    }

    // Push-side multi-input Barrier: two active boundary queues feed one passive
    // join. The join must fire once per matching source generation.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<TwoOutLiveSource>("src", 10));
        auto q0 = pipeline.add(std::make_shared<ActiveBlockingQueue>("q0"));
        auto q1 = pipeline.add(std::make_shared<ActiveBlockingQueue>("q1"));
        auto join = pipeline.add(std::make_shared<Join>("join"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src0", q0, "sink");
        pipeline.link(src, "src1", q1, "sink");
        pipeline.link(q0, "src", join, "in0");
        pipeline.link(q1, "src", join, "in1");
        pipeline.link(join, "out", sink, "in");

        std::stop_source source;
        (void)pipeline.run_threaded(source.get_token());

        const std::vector<std::uint32_t> expected{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u};
        if (!expect(sink_element->slot1 == expected,
                    "push-side Barrier must fire once per matched generation (1..10)")) {
            return 1;
        }
    }

    // Push-side Barrier realign: one active branch drops even generations before
    // the Queue, the other branch carries all generations. The collector must
    // discard orphaned even heads and fire on 1,3,5,7,9.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<TwoOutLiveSource>("src", 10));
        auto odd = pipeline.add(std::make_shared<DropEvenAncestor>("odd"));
        auto all = pipeline.add(std::make_shared<Passthrough>("all"));
        auto q_odd = pipeline.add(std::make_shared<ActiveBlockingQueue>("q_odd"));
        auto q_all = pipeline.add(std::make_shared<ActiveBlockingQueue>("q_all"));
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
                    "push-side Barrier must realign dropped generations to 1,3,5,7,9")) {
            return 1;
        }
    }

    // Push-side Held: a static one-run key behind an active Queue is retained and
    // reused for every live trace.
    {
        leakflow::Pipeline pipeline;
        auto traces = pipeline.add(std::make_shared<LiveSource>("traces", 8));
        auto key = pipeline.add(std::make_shared<OneRunSource>("key"));
        auto q_traces = pipeline.add(std::make_shared<ActiveBlockingQueue>("q_traces"));
        auto q_key = pipeline.add(std::make_shared<ActiveBlockingQueue>("q_key"));
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

        const std::vector<std::uint32_t> expected{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
        if (!expect(sink_element->slot1 == expected,
                    "push-side Held must reuse the static key for every live trace")) {
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

    // Thread boundaries must provide a runtime capability, not just the descriptor
    // flag. This keeps future Queue-like elements honest and gives plugin authors a
    // clear error.
    {
        leakflow::Pipeline pipeline;
        auto src = pipeline.add(std::make_shared<LiveSource>("src", 1));
        auto boundary = pipeline.add(std::make_shared<DescriptorOnlyBoundary>("bad_boundary"));
        auto sink_element = std::make_shared<CountingSink>("sink");
        auto sink = pipeline.add(sink_element);
        pipeline.link(src, "src", boundary, "sink");
        pipeline.link(boundary, "src", sink, "in");

        bool threw = false;
        std::string message;
        try {
            std::stop_source source;
            (void)pipeline.run_threaded(source.get_token());
        } catch (const std::invalid_argument &error) {
            threw = true;
            message = error.what();
        }
        if (!expect(threw && message.find("does not implement ThreadBoundaryRuntime") != std::string::npos,
                    "descriptor-only thread boundary did not fail with a clear runtime-capability error")) {
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
        } catch (const std::exception& error) {
            threw = true;
            message = error.what();
        }
        if (!expect(threw && message.find("element 'throwing' failed with input pad 'in': detailed threaded failure")
                                 != std::string::npos,
                "threaded runner did not rethrow the originating exception with element/pad context")) {
            return 1;
        }
        if (!expect(observer->message.find("element 'throwing' failed with input pad 'in': detailed threaded failure")
                                 != std::string::npos,
                "threaded runner did not expose the originating element/pad error to observers")) {
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
