#pragma once

#include "leakflow/core/property.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace leakflow {

// Telemetry reuses the same typed value container as properties, but lives in a
// separate semantic space: runtime observations/counters/measurements, not user
// configuration and never a replay/invalidation trigger.
using TelemetryValue = PropertyValue;

// Two categories of telemetry share one plumbing path but two gates (see the
// profiling design doc, docs/design/profiling.md):
//
// - Size: cheap monitoring counters/gauges (e.g. Queue depth). Gated by the
//   runtime telemetry switch; safe to leave on (default with --graph).
// - Duration: timing measurements for performance profiling. Gated separately by
//   the profiling switch, because reading the clock has a (small) cost and a
//   self-observer effect you do not want on every monitored run.
enum class TelemetryKind {
    Size,
    Duration,
};

struct TelemetrySpec {
    TelemetrySpec(std::string name,
        TelemetryValue value_type,
        std::string description = {},
        std::string unit = {},
        std::string value_hint = {})
        : name(std::move(name))
        , value_type(std::move(value_type))
        , description(std::move(description))
        , unit(std::move(unit))
        , value_hint(std::move(value_hint))
    {
    }

    std::string name;
    TelemetryValue value_type;
    std::string description;
    std::string unit;
    std::string value_hint;
    TelemetryKind kind = TelemetryKind::Size;
};

struct TelemetrySnapshot {
    std::string name;
    TelemetryValue value;
    std::string description;
    std::string unit;
    std::string value_hint;
    TelemetryKind kind = TelemetryKind::Size;
};

// Aggregate timing report for one duration channel over a whole run. This is the
// row the profile summary table renders and the data the --graph panel could
// surface; it carries no per-event spans (those go to the trace sink).
struct TelemetryDurationReport {
    std::string name;
    std::string description;
    std::string unit = "ns";
    std::uint64_t count = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t min_ns = 0;
    std::uint64_t max_ns = 0;

    [[nodiscard]] std::uint64_t average_ns() const noexcept { return count == 0 ? 0 : total_ns / count; }
};

void validate_telemetry_spec(const TelemetrySpec &spec);

// Convenience builder for a duration (profiling) telemetry spec. The value type
// is an int64 nanosecond exemplar; the unit defaults to nanoseconds.
[[nodiscard]] TelemetrySpec make_duration_telemetry_spec(std::string name,
    std::string description = {},
    std::string unit = "ns");

// Runtime telemetry bookkeeping is controlled by the framework. Elements may own
// counters/gauges, but they should not have to branch on frontend policy. These
// primitives become cheap no-ops when their bound runtime switch is disabled.
class RuntimeTelemetrySwitch {
public:
    void set_enabled(bool enabled) noexcept { enabled_.store(enabled, std::memory_order_relaxed); }
    [[nodiscard]] bool enabled() const noexcept { return enabled_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> enabled_ = true;
};

class RuntimeTelemetrySizeGauge {
public:
    void bind(const RuntimeTelemetrySwitch &runtime) noexcept { runtime_ = &runtime; }
    void reset(std::size_t value = 0) noexcept { value_.store(value, std::memory_order_relaxed); }
    void set(std::size_t value) noexcept
    {
        if (enabled()) {
            value_.store(value, std::memory_order_relaxed);
        }
    }
    [[nodiscard]] std::size_t value() const noexcept { return value_.load(std::memory_order_relaxed); }

private:
    [[nodiscard]] bool enabled() const noexcept { return runtime_ == nullptr || runtime_->enabled(); }

    const RuntimeTelemetrySwitch *runtime_ = nullptr;
    std::atomic<std::size_t> value_ = 0;
};

class RuntimeTelemetrySizeCounter {
public:
    void bind(const RuntimeTelemetrySwitch &runtime) noexcept { runtime_ = &runtime; }
    void reset(std::size_t value = 0) noexcept { value_.store(value, std::memory_order_relaxed); }
    void increment(std::size_t delta = 1) noexcept
    {
        if (enabled()) {
            value_.fetch_add(delta, std::memory_order_relaxed);
        }
    }
    [[nodiscard]] std::size_t value() const noexcept { return value_.load(std::memory_order_relaxed); }

private:
    [[nodiscard]] bool enabled() const noexcept { return runtime_ == nullptr || runtime_->enabled(); }

    const RuntimeTelemetrySwitch *runtime_ = nullptr;
    std::atomic<std::size_t> value_ = 0;
};

class RuntimeTelemetrySizePeak {
public:
    void bind(const RuntimeTelemetrySwitch &runtime) noexcept { runtime_ = &runtime; }
    void reset(std::size_t value = 0) noexcept { value_.store(value, std::memory_order_relaxed); }
    void observe(std::size_t value) noexcept
    {
        if (!enabled()) {
            return;
        }
        auto observed_peak = value_.load(std::memory_order_relaxed);
        while (observed_peak < value &&
               !value_.compare_exchange_weak(
                   observed_peak, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }
    [[nodiscard]] std::size_t value() const noexcept { return value_.load(std::memory_order_relaxed); }

private:
    [[nodiscard]] bool enabled() const noexcept { return runtime_ == nullptr || runtime_->enabled(); }

    const RuntimeTelemetrySwitch *runtime_ = nullptr;
    std::atomic<std::size_t> value_ = 0;
};

// Lock-free timing accumulator: count, total, min, and max of observed durations
// in nanoseconds. Like the size primitives, it becomes a no-op when its bound
// switch is disabled, and it is safe to observe from one thread while a reader
// (the report) reads the atomics. min/max use the same CAS loop as the size peak.
//
// Bind this to the element's PROFILING switch (not the size telemetry switch):
// timing is opt-in because the clock read has a cost and a self-observer effect.
class RuntimeTelemetryDurationStat {
public:
    void bind(const RuntimeTelemetrySwitch &runtime) noexcept { runtime_ = &runtime; }

    void reset() noexcept
    {
        count_.store(0, std::memory_order_relaxed);
        total_ns_.store(0, std::memory_order_relaxed);
        min_ns_.store(UINT64_MAX, std::memory_order_relaxed);
        max_ns_.store(0, std::memory_order_relaxed);
    }

    void observe(std::chrono::nanoseconds elapsed) noexcept
    {
        if (!enabled()) {
            return;
        }
        const auto value = elapsed.count() < 0
            ? std::uint64_t{0}
            : static_cast<std::uint64_t>(elapsed.count());
        count_.fetch_add(1, std::memory_order_relaxed);
        total_ns_.fetch_add(value, std::memory_order_relaxed);

        auto observed_min = min_ns_.load(std::memory_order_relaxed);
        while (value < observed_min &&
               !min_ns_.compare_exchange_weak(
                   observed_min, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
        auto observed_max = max_ns_.load(std::memory_order_relaxed);
        while (observed_max < value &&
               !max_ns_.compare_exchange_weak(
                   observed_max, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return count_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t total_ns() const noexcept { return total_ns_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t min_ns() const noexcept
    {
        return count() == 0 ? 0 : min_ns_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t max_ns() const noexcept { return max_ns_.load(std::memory_order_relaxed); }

    [[nodiscard]] TelemetryDurationReport report(std::string name, std::string description = {},
        std::string unit = "ns") const
    {
        return TelemetryDurationReport{
            .name = std::move(name),
            .description = std::move(description),
            .unit = std::move(unit),
            .count = count(),
            .total_ns = total_ns(),
            .min_ns = min_ns(),
            .max_ns = max_ns(),
        };
    }

private:
    [[nodiscard]] bool enabled() const noexcept { return runtime_ == nullptr || runtime_->enabled(); }

    const RuntimeTelemetrySwitch *runtime_ = nullptr;
    std::atomic<std::uint64_t> count_ = 0;
    std::atomic<std::uint64_t> total_ns_ = 0;
    std::atomic<std::uint64_t> min_ns_ = UINT64_MAX;
    std::atomic<std::uint64_t> max_ns_ = 0;
};

} // namespace leakflow
