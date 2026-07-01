#include "leakflow/plugins/core/queue.hpp"

#include "core_plugin_constants.hpp"
#include "property_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace leakflow::plugins::core {
namespace {

constexpr std::string_view telemetry_size_name = "size";
constexpr std::string_view telemetry_peak_size_name = "peak_size";
constexpr std::string_view telemetry_received_name = "received";
constexpr std::string_view telemetry_emitted_name = "emitted";
constexpr std::string_view telemetry_dropped_name = "dropped";
constexpr std::string_view telemetry_unit_buffers = "buffers";
constexpr std::string_view telemetry_size_description = "current number of buffers held by this queue";
constexpr std::string_view telemetry_peak_size_description = "highest queue depth observed in the current run";
constexpr std::string_view telemetry_received_description =
    "number of buffers accepted into this queue in the current run";
constexpr std::string_view telemetry_emitted_description =
    "number of buffers emitted from this queue in the current run";
constexpr std::string_view telemetry_dropped_description = "number of buffers discarded by this queue in the current run";
constexpr std::string_view telemetry_backpressure_name = "backpressure";
constexpr std::string_view telemetry_starvation_name = "starvation";
constexpr std::string_view telemetry_backpressure_description =
    "time the producer spent blocked on a full queue (downstream too slow)";
constexpr std::string_view telemetry_starvation_description =
    "time the consumer spent blocked on an empty queue (upstream too slow)";

[[nodiscard]] std::int64_t telemetry_value(std::size_t value)
{
    const auto clamped = std::min<std::size_t>(
        value, static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
    return static_cast<std::int64_t>(clamped);
}

[[nodiscard]] TelemetrySpec queue_telemetry_spec(
    std::string_view name,
    std::string_view description)
{
    return TelemetrySpec(
        std::string(name),
        std::int64_t{0},
        std::string(description),
        std::string(telemetry_unit_buffers));
}

[[nodiscard]] TelemetrySnapshot queue_telemetry_snapshot(
    std::string_view name,
    std::string_view description,
    std::size_t value)
{
    return TelemetrySnapshot{
        .name = std::string(name),
        .value = telemetry_value(value),
        .description = std::string(description),
        .unit = std::string(telemetry_unit_buffers),
    };
}

} // namespace

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
            PropertySpec("drop_oldest", false, "drop the oldest buffer when the queue is full"),
        },
        .telemetry_specs = {
            queue_telemetry_spec(telemetry_size_name, telemetry_size_description),
            queue_telemetry_spec(telemetry_peak_size_name, telemetry_peak_size_description),
            queue_telemetry_spec(telemetry_received_name, telemetry_received_description),
            queue_telemetry_spec(telemetry_emitted_name, telemetry_emitted_description),
            queue_telemetry_spec(telemetry_dropped_name, telemetry_dropped_description),
            // Profiling (duration) channels: the meaningful timing for a thread
            // boundary is wait time, not process time. See docs/design/profiling.md.
            make_duration_telemetry_spec(
                std::string(telemetry_backpressure_name), std::string(telemetry_backpressure_description)),
            make_duration_telemetry_spec(
                std::string(telemetry_starvation_name), std::string(telemetry_starvation_description)),
        },
        .keywords = {"queue", "buffer", "storage"},
        .can_replay = false,
        .provenance_slots = 0,
        .thread_boundary = true,
    };
}

Queue::Queue(std::string name)
    : Element(std::move(name))
{
    telemetry_size_.bind(runtime_telemetry());
    telemetry_peak_size_.bind(runtime_telemetry());
    telemetry_received_.bind(runtime_telemetry());
    telemetry_emitted_.bind(runtime_telemetry());
    telemetry_dropped_.bind(runtime_telemetry());
    configure_from_descriptor(descriptor());
}

void Queue::start()
{
    // Lifecycle reset (Phase 25, D3 clause 2): start() must restore a clean
    // baseline so a stop_all -> start_all -> run_sweep cycle is a true reset.
    buffers_.clear();
    reset_telemetry();
}

std::optional<Buffer> Queue::process(std::optional<Buffer> input)
{
    if (input) {
        const auto max_size = static_cast<std::size_t>(
            std::max<std::int64_t>(1, int_property_or(*this, "max_size", 1)));
        if (buffers_.size() >= max_size) {
            if (!bool_property_or(*this, "drop_oldest", false)) {
                throw std::runtime_error("Queue is full");
            }
            auto record = make_log_record(log::LogLevel::Warning, "element", "queue full; dropped oldest buffer");
            record.fields.emplace("max_size", std::to_string(max_size));
            leakflow::log::write(std::move(record));
            buffers_.pop_front();
            telemetry_dropped_.increment();
        }
        buffers_.push_back(std::move(*input));
        record_telemetry_received(buffers_.size(), false);
    }

    if (buffers_.empty()) {
        record_telemetry_size(0);
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "queue is empty"));
        return std::nullopt;
    }

    auto output = std::move(buffers_.front());
    buffers_.pop_front();
    record_telemetry_emitted(buffers_.size());
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

void Queue::prepare_thread_boundary_runtime(std::mutex *)
{
    const auto max_size = static_cast<std::size_t>(
        std::max<std::int64_t>(1, int_property_or(*this, "max_size", 1)));
    const bool drop_oldest = bool_property_or(*this, "drop_oldest", false);
    runtime_queue_ = std::make_unique<BufferQueue>(
        max_size, drop_oldest ? QueueDropPolicy::DropOldest : QueueDropPolicy::Block);
    reset_telemetry();
}

void Queue::clear_thread_boundary_runtime() noexcept
{
    try {
        if (runtime_queue_) {
            runtime_queue_->close();
        }
        if (active_worker_.joinable()) {
            active_worker_ = std::jthread{};
        }
        runtime_queue_.reset();
        record_telemetry_size(0);
    } catch (...) {
    }
}

bool Queue::boundary_push(Buffer buffer, std::stop_token stop)
{
    // push blocks under backpressure (Block policy, full queue); the wait time is
    // the "downstream too slow" signal. No-op scope when profiling is off.
    const auto scope = profile_scope(telemetry_backpressure_name);
    const auto push = runtime_queue().push_with_status(std::move(buffer), stop);
    if (push.accepted) {
        record_telemetry_received(push.size_after, push.dropped);
    } else {
        record_telemetry_size(push.size_after);
    }
    return push.accepted;
}

BufferQueue::Pull Queue::boundary_pull(std::stop_token stop)
{
    // pull blocks while the queue is empty; the wait time is the "upstream too
    // slow"/starvation signal.
    const auto scope = profile_scope(telemetry_starvation_name);
    auto pull = runtime_queue().pull(stop);
    if (pull.buffer) {
        record_telemetry_emitted(pull.size_after);
    } else {
        record_telemetry_size(pull.size_after);
    }
    return pull;
}

BufferQueue::Pull Queue::boundary_try_pull()
{
    auto pull = runtime_queue().try_pull();
    if (pull.buffer) {
        record_telemetry_emitted(pull.size_after);
    } else {
        record_telemetry_size(pull.size_after);
    }
    return pull;
}

void Queue::boundary_close()
{
    runtime_queue().close();
    record_telemetry_size(runtime_queue().size());
}

std::vector<TelemetrySnapshot> Queue::boundary_runtime_telemetry()
{
    if (runtime_queue_ != nullptr) {
        record_telemetry_size(runtime_queue_->size());
    }
    return telemetry_snapshot();
}

void Queue::start_active(RuntimeContext &context)
{
    if (active_worker_.joinable()) {
        stop_active();
    }
    active_worker_ = std::jthread([this, &context](std::stop_token local_stop) {
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
            context.report_error(*this, "unknown Queue active source failure");
        }
    });
}

void Queue::wait_active()
{
    if (active_worker_.joinable()) {
        active_worker_.join();
    }
}

void Queue::stop_active() noexcept
{
    try {
        if (runtime_queue_) {
            runtime_queue_->close();
        }
    } catch (...) {
    }
    if (active_worker_.joinable()) {
        active_worker_ = std::jthread{};
    }
}

void Queue::reset_telemetry()
{
    telemetry_size_.reset();
    telemetry_peak_size_.reset();
    telemetry_received_.reset();
    telemetry_emitted_.reset();
    telemetry_dropped_.reset();
}

void Queue::record_telemetry_size(std::size_t size)
{
    telemetry_size_.set(size);
    telemetry_peak_size_.observe(size);
}

void Queue::record_telemetry_received(std::size_t size_after, bool dropped)
{
    telemetry_received_.increment();
    if (dropped) {
        telemetry_dropped_.increment();
    }
    record_telemetry_size(size_after);
}

void Queue::record_telemetry_emitted(std::size_t size_after)
{
    telemetry_emitted_.increment();
    record_telemetry_size(size_after);
}

Queue::TelemetryCounters Queue::telemetry_counters() const
{
    return TelemetryCounters{
        .size = telemetry_size_.value(),
        .peak_size = telemetry_peak_size_.value(),
        .received = telemetry_received_.value(),
        .emitted = telemetry_emitted_.value(),
        .dropped = telemetry_dropped_.value(),
    };
}

std::vector<TelemetrySnapshot> Queue::telemetry_snapshot() const
{
    const auto counters = telemetry_counters();
    std::vector<TelemetrySnapshot> snapshots = {
        queue_telemetry_snapshot(telemetry_size_name, telemetry_size_description, counters.size),
        queue_telemetry_snapshot(telemetry_peak_size_name, telemetry_peak_size_description, counters.peak_size),
        queue_telemetry_snapshot(telemetry_received_name, telemetry_received_description, counters.received),
        queue_telemetry_snapshot(telemetry_emitted_name, telemetry_emitted_description, counters.emitted),
        queue_telemetry_snapshot(telemetry_dropped_name, telemetry_dropped_description, counters.dropped),
    };

    // Append the recorded wait-time (profiling) channels as ms / kind Duration, so
    // they show in the graph's Profile section and the --print-profile table.
    for (const auto &report : duration_reports()) {
        snapshots.push_back(TelemetrySnapshot{
            .name = report.name,
            .value = static_cast<double>(report.total_ns) / 1'000'000.0,
            .description = report.description,
            .unit = "ms",
            .value_hint = {},
            .kind = TelemetryKind::Duration,
        });
    }
    return snapshots;
}

BufferQueue &Queue::runtime_queue()
{
    if (runtime_queue_ == nullptr) {
        throw std::logic_error("Queue boundary runtime is not prepared");
    }
    return *runtime_queue_;
}

} // namespace leakflow::plugins::core
