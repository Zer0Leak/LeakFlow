#include "leakflow/plugins/core/queue.hpp"

#include "core_plugin_constants.hpp"
#include "property_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace leakflow::plugins::core {

ElementDescriptor Queue::descriptor()
{
    return {
        .type_name = "Queue",
        .klass = "PassThrough/Queue",
        .purpose = "store buffers synchronously between linked elements",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(core_pad_caps_type)),
        },
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(core_pad_caps_type)),
        },
        .property_specs = {
            PropertySpec(
                "max_size",
                std::int64_t{1},
                "maximum number of buffers to keep before dropping the oldest",
                "buffers",
                IntRangeConstraint{1, 1024}),
            PropertySpec("drop_oldest", true, "drop the oldest buffer when the queue is full"),
        },
        .keywords = {"queue", "buffer", "storage"},
        .can_replay = false,
        .thread_boundary = true,
    };
}

Queue::Queue(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

void Queue::start()
{
    // Lifecycle reset (Phase 25, D3 clause 2): start() must restore a clean
    // baseline so a stop_all -> start_all -> run_sweep cycle is a true reset.
    buffers_.clear();
}

std::optional<Buffer> Queue::process(std::optional<Buffer> input)
{
    if (input) {
        const auto max_size = static_cast<std::size_t>(
            std::max<std::int64_t>(1, int_property_or(*this, "max_size", 1)));
        if (buffers_.size() >= max_size) {
            if (!bool_property_or(*this, "drop_oldest", true)) {
                throw std::runtime_error("Queue is full");
            }
            auto record = make_log_record(log::LogLevel::Warning, "element", "queue full; dropped oldest buffer");
            record.fields.emplace("max_size", std::to_string(max_size));
            leakflow::log::write(std::move(record));
            buffers_.pop_front();
        }
        buffers_.push_back(std::move(*input));
    }

    if (buffers_.empty()) {
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "queue is empty"));
        return std::nullopt;
    }

    auto output = std::move(buffers_.front());
    buffers_.pop_front();
    auto record = make_log_record(log::LogLevel::Debug, "element", "forwarded queued buffer");
    record.fields.emplace("depth", std::to_string(buffers_.size()));
    leakflow::log::write(std::move(record));
    return output;
}

bool Queue::can_replay() const
{
    return false;
}

std::size_t Queue::depth() const
{
    return buffers_.size();
}

} // namespace leakflow::plugins::core
