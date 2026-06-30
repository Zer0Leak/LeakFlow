#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/buffer_queue.hpp"
#include "leakflow/core/pipeline_observer.hpp"
#include "leakflow/core/telemetry.hpp"

#include <mutex>
#include <stop_token>
#include <vector>

namespace leakflow {

// Runtime contract for elements that cut the graph into threaded segments.
//
// The descriptor's thread_boundary flag is the declarative graph property; this
// interface is the runtime capability. The pipeline remains the lifecycle and
// routing authority, but it no longer owns the boundary queue storage for Queue-like
// elements. A boundary element owns its handoff state and exposes push/pull/close.
class ThreadBoundaryRuntime {
public:
    virtual ~ThreadBoundaryRuntime() = default;

    // Called by Pipeline::run_threaded after element start() and before segment
    // threads are launched. The pointer may be null. When non-null, implementations
    // should use it to serialize read-only runtime property updates with the graph
    // UI/session property reader.
    virtual void prepare_thread_boundary_runtime(std::mutex *property_mutex) = 0;

    // Called after segment threads have joined. Must not throw: it is used during
    // both normal shutdown and failure cleanup to wake/close/reset runtime state.
    virtual void clear_thread_boundary_runtime() noexcept = 0;

    // Producer side. Returns false only when the buffer was not accepted because of
    // cooperative stop or boundary shutdown.
    [[nodiscard]] virtual bool boundary_push(Buffer buffer, std::stop_token stop) = 0;

    // Consumer side. See BufferQueue::Pull for Data / NoData / EOS states.
    [[nodiscard]] virtual BufferQueue::Pull boundary_pull(std::stop_token stop) = 0;
    [[nodiscard]] virtual BufferQueue::Pull boundary_try_pull() = 0;

    // Producer EOS. Consumers drain already-buffered data, then observe EOS.
    virtual void boundary_close() = 0;

    // Optional runtime properties owned by the boundary element. Older hook kept
    // for compatibility; new runtime observations should use
    // boundary_runtime_telemetry().
    [[nodiscard]] virtual std::vector<PipelinePropertySnapshot> boundary_runtime_properties() { return {}; }

    // Optional runtime telemetry owned by the boundary element (for example
    // Queue size/peak/received/emitted/dropped counters). The threaded runner samples
    // these snapshots at UI cadence, coalesces unchanged values, and publishes
    // copied TelemetryChanged events so graph hover/pinned panels stay current
    // without per-buffer UI events.
    [[nodiscard]] virtual std::vector<TelemetrySnapshot> boundary_runtime_telemetry() { return {}; }
};

} // namespace leakflow
