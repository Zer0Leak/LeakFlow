#pragma once

#include "leakflow/core/telemetry.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace leakflow {

// One completed timing span for the Chrome Trace Event format. Times are in
// microseconds relative to the owning sink's epoch. See docs/design/profiling.md.
struct TelemetryTraceSpan {
    std::string name;     // e.g. "aesleakage0.process" or "aesleakage0.hamming_weight"
    std::string category; // grouping; the element instance name
    std::uint64_t thread_id = 0;
    // Nanoseconds relative to the sink epoch. Exported as fractional microseconds
    // so sub-microsecond and nested scopes stay strictly nested (Perfetto drops
    // overlapping complete events that collapse onto the same integer timestamp).
    std::uint64_t timestamp_ns = 0;
    std::uint64_t duration_ns = 0;
};

// Thread-safe recorder of per-event timing spans. Only created when the user asks
// for a trace file (the heavy per-event path); the cheap aggregate stats record
// independently. Exports the Chrome Trace Event JSON consumed by chrome://tracing
// and Perfetto, so LeakFlow needs no flame-graph GUI of its own.
class TelemetryTraceSink {
public:
    TelemetryTraceSink() : epoch_(std::chrono::steady_clock::now()) {}

    void add_complete(std::string name, std::string category,
        std::chrono::steady_clock::time_point begin,
        std::chrono::steady_clock::time_point end);

    [[nodiscard]] bool empty() const;
    [[nodiscard]] std::size_t span_count() const;

    // Chrome Trace Event JSON: a top-level array of "X" (complete) events.
    [[nodiscard]] std::string to_chrome_json() const;

private:
    std::chrono::steady_clock::time_point epoch_;
    mutable std::mutex mutex_;
    std::vector<TelemetryTraceSpan> spans_;
};

// RAII scope timer. Records elapsed wall time into a duration stat on destruction,
// and (when a trace sink is set) also appends a complete span. Inactive when the
// stat pointer is null (profiling off), so the cost is a single null check.
//
// Reused by the executor for per-element process timing and by elements for
// opt-in internal op scopes (Element::profile_scope).
class RuntimeTelemetryScopedTimer {
public:
    RuntimeTelemetryScopedTimer(RuntimeTelemetryDurationStat *stat, TelemetryTraceSink *sink,
        std::string name, std::string category)
        : stat_(stat), sink_(sink)
    {
        active_ = stat_ != nullptr || sink_ != nullptr;
        if (active_) {
            name_ = std::move(name);
            category_ = std::move(category);
            begin_ = std::chrono::steady_clock::now();
        }
    }

    RuntimeTelemetryScopedTimer(const RuntimeTelemetryScopedTimer &) = delete;
    RuntimeTelemetryScopedTimer &operator=(const RuntimeTelemetryScopedTimer &) = delete;

    ~RuntimeTelemetryScopedTimer()
    {
        if (!active_) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        if (stat_ != nullptr) {
            stat_->observe(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin_));
        }
        if (sink_ != nullptr) {
            sink_->add_complete(std::move(name_), std::move(category_), begin_, end);
        }
    }

private:
    RuntimeTelemetryDurationStat *stat_ = nullptr;
    TelemetryTraceSink *sink_ = nullptr;
    bool active_ = false;
    std::string name_;
    std::string category_;
    std::chrono::steady_clock::time_point begin_;
};

} // namespace leakflow
