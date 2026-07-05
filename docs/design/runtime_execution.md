# Runtime Execution Model

Status: current design of record for LeakFlow runtime ownership and future active
elements.

LeakFlow is moving toward a GStreamer-like runtime model, but with LeakFlow-sized
interfaces:

- **Elements own behavior and local runtime state.**
- **Pads are the public dataflow surface.**
- **`Pipeline` owns orchestration:** lifecycle, routing, provenance, observer
  events, error propagation, and control-plane policy.

This is intentionally a hybrid. Most side-channel analysis elements remain passive
and deterministic:

```text
process_pads(inputs) -> outputs
```

Examples: `AesLeakage`, `AesLeakageHypothesis`, `CpaAttack`, `AttackStats`,
`PearsonCorrelator`, `PoiSelect`, annotations, summaries.

Only elements that naturally drive or decouple a stream should become runtime-aware:

```text
sources, Queue-like boundaries, capture hardware, file/watch/network sources,
and selected UI bridges
```

## Boundary runtime: implemented now

`ThreadBoundaryRuntime` is the first concrete runtime capability:

```cpp
class ThreadBoundaryRuntime {
public:
    virtual void prepare_thread_boundary_runtime(std::mutex *property_mutex) = 0;
    virtual void clear_thread_boundary_runtime() noexcept = 0;
    virtual bool boundary_push(Buffer buffer, std::stop_token stop) = 0;
    virtual BufferQueue::Pull boundary_pull(std::stop_token stop) = 0;
    virtual BufferQueue::Pull boundary_try_pull() = 0;
    virtual void boundary_close() = 0;
};
```

The descriptor flag:

```cpp
descriptor.thread_boundary = true;
```

is the graph declaration. `ThreadBoundaryRuntime` is the runtime capability. A
thread-boundary element must implement both.

`Queue` now owns its runtime `BufferQueue`:

```text
Queue
  owns BufferQueue
  owns first-class telemetry updates
  exposes push / pull / try_pull / close through ThreadBoundaryRuntime
```

The pipeline no longer creates a separate `BufferQueue` for `Queue`. It only asks
the boundary element to prepare, push, pull, close, and clear. During a threaded
run, the pipeline samples every boundary's copied runtime telemetry snapshots at
UI cadence, coalesces unchanged values, and emits changed values as
`TelemetryChanged` events; this keeps graph hover/pinned topology data such as
`Queue` telemetry `size` and `peak_size` live without per-buffer UI events.
Runtime telemetry is optional: when disabled for a run, the threaded runner does
not start the sampler/publisher, and framework telemetry primitives become
no-ops so elements such as `Queue` do not need local enable/disable branches.

This keeps the existing segment runner but moves ownership to the element:

```text
Pipeline segment thread T1 ── boundary_push() ─▶ Queue-owned BufferQueue
Pipeline segment thread T2 ◀─ boundary_pull() ── Queue-owned BufferQueue
```

## RuntimeContext and ActiveElement: active sources and active boundaries

Selected source-like and boundary-like elements can own tasks that push from their
own threads. The narrow API for that is `RuntimeContext`:

```cpp
class RuntimeContext {
public:
    virtual bool push(Element &element, std::string_view source_pad, Buffer buffer) = 0;
    virtual void end_of_stream(Element &element, std::string_view source_pad) = 0;
    virtual void report_error(Element &element, std::string message) = 0;
    virtual void safe_point(Element &element) = 0;
    virtual std::stop_token stop_token() const = 0;
    virtual bool stop_requested() const = 0;
    virtual bool is_paused() const = 0;
    virtual void wait_if_paused() = 0;
};
```

`ActiveElement` is the optional capability:

```cpp
class ActiveElement {
public:
    virtual void start_active(RuntimeContext &context) = 0;
    virtual void wait_active() = 0;
    virtual void stop_active() noexcept = 0;
};
```

These interfaces deliberately do **not** expose `Pipeline *` to plugins. An active
element may push from its own pads, report its own errors, and observe lifecycle
state; it may not mutate topology or other elements.

The implemented runtime path is intentionally narrow but useful:

```text
ActiveElement task
  -> RuntimeContext::push(element, "src", buffer)
  -> pipeline preserves existing provenance and stamps the active element's own
     slot when it has one
  -> applies output metadata annotations
  -> emits BufferObserved
  -> pushes into a downstream ThreadBoundaryRuntime, or
  -> feeds passive downstream elements through the runtime push collector
```

So active source segments are supported when their outputs feed Queue-like
boundaries, and active boundary elements can push into passive downstream chains.

The push collector handles one-input chains, Queue → passive chain → Queue, and
multi-input passive joins. For joins it keeps one FIFO per input pad, retains
Held/Latest pads, consumes Driving pads, and realigns Driving heads by vector
clock before firing. This mirrors the existing pull-side Barrier/Held/Latest
logic.

## Active Queue source task

The current Queue shape is:

```text
Queue.sink side:
  receives pushed buffers and enqueues

Queue-owned src task:
  waits for queued buffers
  pushes downstream through RuntimeContext
```

`Queue` implements both `ThreadBoundaryRuntime` and `ActiveElement`. During
`run_threaded`, a consumer segment fed only by active-capable boundary elements,
and with no internal source-like producer of its own, is activated by those
boundary tasks instead of by an old pull-side consumer segment. If the segment
contains a passive source such as `TorchFileSrc` feeding a later join, the normal
consumer segment thread still runs so that source is executed.
The runner starts active tasks, waits for natural completion with `wait_active()`,
then tears them down with `stop_active()` during cleanup or failure. In
graph/session runs, active tasks call `RuntimeContext::safe_point(element)` so
pause and runtime property changes still happen at between-buffer points on the
thread that reads the element's properties.

`Queue` claims zero provenance slots. It is a pass-through boundary: the vector
clock rides through it untouched.

The pull-side segment runner remains as a compatibility fallback for consumer
segments whose input boundaries are not active-capable, and for mixed consumer
segments that need to execute internal passive sources.

## Push/pull scheduling lesson from GStreamer

GStreamer chooses push or pull at pad activation time with scheduling queries. A
file source may be push-active or pull-passive depending on downstream needs; a
live capture source is usually push-active. LeakFlow should eventually copy that
idea at the concept level:

```text
upstream advertises capabilities
downstream declares whether it can/wants to drive
runtime activates the compatible mode
```

For now, LeakFlow's implemented runtime mode is:

```text
offline/passive sweeps
live segment threading
Queue-owned boundary handoff
active source task -> RuntimeContext -> Queue boundary
```
