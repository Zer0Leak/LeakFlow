// Latest join policy (S11.1): a primary input drives; a secondary input is sampled
// -- each fire pairs the primary with the NEWEST secondary buffer, dropping
// intermediates and never waiting. The join fires once per primary buffer.

#include "leakflow/core/element.hpp"
#include "leakflow/core/active_element.hpp"
#include "leakflow/core/buffer_queue.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/runtime_context.hpp"
#include "leakflow/core/thread_boundary_runtime.hpp"

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

// A live source that emits `total` buffers tagged 1..total, optionally paced.
class TaggedSource final : public leakflow::Element {
public:
    TaggedSource(std::string name, int total, int delay_ms)
        : Element(std::move(name)), total_(total), delay_ms_(delay_ms)
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "TaggedSource";
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
        buffer.set_metadata("tag", std::to_string(cursor_));
        if (delay_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
        }
        return buffer;
    }

    [[nodiscard]] bool at_end_of_stream() const override { return cursor_ >= total_; }
    void stop() override { cursor_ = 0; }

private:
    int total_;
    int delay_ms_;
    int cursor_ = 0;
};

class BlockingQueue final
    : public leakflow::Element
    , public leakflow::ThreadBoundaryRuntime
    , public leakflow::ActiveElement {
public:
    BlockingQueue(std::string name, std::int64_t capacity)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "BlockingQueue";
        descriptor.klass = "PassThrough/Queue";
        descriptor.input_pads = {leakflow::Pad("sink", leakflow::PadDirection::Input, generic())};
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, generic())};
        descriptor.property_specs = {
            leakflow::PropertySpec("max_size", capacity, "capacity"),
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
                context.report_error(*this, "unknown active latest-test queue failure");
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

// A two-input join carrying a `policy` property; its output records both inputs'
// tags so the test can see which secondary buffer each fire paired with.
class Combine final : public leakflow::Element {
public:
    Combine(std::string name, std::string policy)
        : Element(std::move(name))
    {
        add_input_pad(leakflow::Pad("in0", leakflow::PadDirection::Input, generic()));
        add_input_pad(leakflow::Pad("in1", leakflow::PadDirection::Input, generic()));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, generic()));
        add_property(leakflow::PropertySpec("policy", std::move(policy), "join policy"));
    }

    leakflow::ElementOutputs process_pads(leakflow::ElementInputs inputs) override
    {
        const auto tag_of = [&](const char* pad) -> std::string {
            const auto found = inputs.find(pad);
            if (found == inputs.end() || !found->second) {
                return {};
            }
            const auto meta = found->second->metadata().find("tag");
            return meta != found->second->metadata().end() ? meta->second : std::string();
        };
        leakflow::Buffer buffer(generic());
        buffer.set_metadata("primary", tag_of("in0"));
        buffer.set_metadata("secondary", tag_of("in1"));
        leakflow::ElementOutputs outputs;
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
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, generic()));
    }
    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            const auto &meta = input->metadata();
            primary.push_back(meta.count("primary") ? meta.at("primary") : std::string());
            secondary.push_back(meta.count("secondary") ? meta.at("secondary") : std::string());
        }
        return std::nullopt;
    }
    std::vector<std::string> primary;
    std::vector<std::string> secondary;
};

} // namespace

int main()
{
    // Primary: 3 buffers, 25 ms apart. Secondary: 30 buffers, instant -> all land in
    // its (large) queue long before each slow primary fire. With policy=latest the
    // join fires once per primary (3 times), each sampling the newest secondary (30).
    leakflow::Pipeline pipeline;
    auto primary = pipeline.add(std::make_shared<TaggedSource>("primary", 3, 25));
    auto q_primary = pipeline.add(std::make_shared<BlockingQueue>("qp", 8));
    auto secondary = pipeline.add(std::make_shared<TaggedSource>("secondary", 30, 0));
    auto q_secondary = pipeline.add(std::make_shared<BlockingQueue>("qs", 64));
    auto combine = pipeline.add(std::make_shared<Combine>("combine", "latest"));
    auto sink_element = std::make_shared<RecordingSink>("sink");
    auto sink = pipeline.add(sink_element);
    pipeline.link(primary, "src", q_primary, "sink");
    pipeline.link(secondary, "src", q_secondary, "sink");
    pipeline.link(q_primary, "src", combine, "in0");
    pipeline.link(q_secondary, "src", combine, "in1");
    pipeline.link(combine, "out", sink, "in");

    std::stop_source stop;
    (void)pipeline.run_threaded(stop.get_token());

    if (!expect(sink_element->primary.size() == 3, "Latest must fire once per primary buffer (3)")) {
        std::cerr << "  fired " << sink_element->primary.size() << " times\n";
        return 1;
    }
    const std::vector<std::string> expected_primary{"1", "2", "3"};
    if (!expect(sink_element->primary == expected_primary, "primary tags must be 1,2,3")) {
        return 1;
    }
    bool secondary_newest = true;
    for (const auto &tag : sink_element->secondary) {
        secondary_newest = secondary_newest && tag == "30";
    }
    if (!expect(secondary_newest, "each fire must sample the NEWEST secondary (30), dropping intermediates")) {
        for (const auto &tag : sink_element->secondary) {
            std::cerr << "  secondary=" << tag << '\n';
        }
        return 1;
    }

    return 0;
}
