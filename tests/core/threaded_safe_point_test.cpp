// Safe-point control plane (S11.5): a property change submitted to the session
// WHILE a threaded run is streaming must forward-apply on a subsequent buffer --
// not stall until end-of-stream. Each segment thread drains its own commands at a
// between-buffer safe point, so the write happens on the same thread that reads the
// property in process() (no data race).

#include "leakflow/core/element.hpp"
#include "leakflow/core/active_element.hpp"
#include "leakflow/core/buffer_queue.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/core/runtime_context.hpp"
#include "leakflow/core/thread_boundary_runtime.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
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

// A live source that stamps each emitted buffer with the CURRENT value of its
// "tag" property, paced slightly so a peer thread can edit it mid-stream.
class TaggedSource final : public leakflow::Element {
public:
    TaggedSource(std::string name, int total)
        : Element(std::move(name)), total_(total)
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "TaggedSource";
        descriptor.klass = "Source/Live";
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.property_specs = {leakflow::PropertySpec("tag", std::int64_t{0}, "tag stamped on each buffer")};
        descriptor.live_source = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        if (cursor_ >= total_) {
            return std::nullopt;
        }
        ++cursor_;
        const auto tag = property_as<std::int64_t>("tag").value_or(0);
        leakflow::Buffer buffer(generic());
        buffer.set_metadata("tag", std::to_string(tag));
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // pace for mid-stream editing
        return buffer;
    }

    [[nodiscard]] bool at_end_of_stream() const override { return cursor_ >= total_; }
    void stop() override { cursor_ = 0; }

private:
    int total_;
    int cursor_ = 0;
};

class BlockingQueue final
    : public leakflow::Element
    , public leakflow::ThreadBoundaryRuntime
    , public leakflow::ActiveElement {
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
            leakflow::PropertySpec("max_size", std::int64_t{8}, "capacity"),
            leakflow::PropertySpec("drop_oldest", false, "block when full"),
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
                context.report_error(*this, "unknown active safe-point-test queue failure");
            }
        });
    }

    void wait_active() override
    {
        if (worker_.joinable()) {
            worker_.join();
        }
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

private:
    [[nodiscard]] leakflow::BufferQueue &runtime_queue()
    {
        if (runtime_queue_ == nullptr) {
            throw std::logic_error("BlockingQueue runtime is not prepared");
        }
        return *runtime_queue_;
    }

    std::unique_ptr<leakflow::BufferQueue> runtime_queue_;
    std::jthread worker_;
};

class TagSink final : public leakflow::Element {
public:
    explicit TagSink(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            const auto found = input->metadata().find("tag");
            tags.push_back(found != input->metadata().end() ? found->second : std::string());
            count.fetch_add(1);
        }
        return std::nullopt;
    }

    std::atomic<int> count = 0;
    std::vector<std::string> tags;
};

} // namespace

int main()
{
    leakflow::Pipeline pipeline;
    auto src = std::make_shared<TaggedSource>("src", 200);
    auto src_handle = pipeline.add(src);
    auto queue = pipeline.add(std::make_shared<BlockingQueue>("q"));
    auto sink_element = std::make_shared<TagSink>("sink");
    auto sink = pipeline.add(sink_element);
    pipeline.link(src_handle, "src", queue, "sink");
    pipeline.link(queue, "src", sink, "in");

    leakflow::PipelineSession session(std::move(pipeline));
    if (!expect(session.pipeline().should_run_threaded(), "live source + queue must run threaded")) {
        return 1;
    }

    std::stop_source stop;
    session.set_stop_token(stop.get_token());

    // Run the threaded session; submit a tag change once the stream is underway.
    std::thread runner([&session]() { (void)session.run_once(); });

    for (int attempt = 0; attempt < 4000 && sink_element->count.load() < 20; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    session.submit(leakflow::SetPropertyCommand{
        .element_name = "src",
        .property_name = "tag",
        .value = std::int64_t{99},
    });
    runner.join();

    if (!expect(sink_element->count.load() == 200, "the whole stream must arrive (200 buffers)")) {
        return 1;
    }
    if (!expect(!sink_element->tags.empty() && sink_element->tags.front() == "0",
                "early buffers carry the original tag (0)")) {
        return 1;
    }
    if (!expect(sink_element->tags.back() == "99", "the mid-stream tag change must forward-apply before EOS")) {
        return 1;
    }
    // The change took effect partway through -- not only on the final buffer.
    int changed = 0;
    for (const auto& tag : sink_element->tags) {
        if (tag == "99") {
            ++changed;
        }
    }
    if (!expect(changed > 1 && changed < 200, "the change must apply mid-stream, not at the very end or start")) {
        return 1;
    }

    return 0;
}
