#pragma once

#include <cstdint>
#include <string>

namespace leakflow {

class Element;

enum class ProgressStatus {
    Active,
    Completed,
    Cancelled,
};

// Estimated progress from a long-running element, reported from inside process().
struct ElementProgress {
    double fraction = 0.0;   // estimated completion in [0, 1]
    std::string message;     // human-readable stage, e.g. "restart 2/3 - iter 40/100"
    std::uint64_t index = 0; // optional current step (0 when unused)
    std::uint64_t total = 0; // optional total steps (0 when unused)
    ProgressStatus status = ProgressStatus::Active;
};

// Framework-injected push channel for progress, mirroring TelemetryTraceSink: the Pipeline sets
// one on each element, and Element::report_progress forwards to it (throttled) from inside
// process() on the worker thread. The Pipeline's implementation dispatches a ProgressReported
// event on the observer bus, so --graph and any PipelineObserver see it. An element with no sink
// set reports nothing (cheap no-op).
class ProgressSink {
public:
    virtual ~ProgressSink() = default;
    virtual void report(Element& element, const ElementProgress& progress) = 0;
};

} // namespace leakflow
