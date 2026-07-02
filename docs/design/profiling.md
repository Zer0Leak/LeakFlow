# LeakFlow Profiling & Timing Telemetry

Design of record for LeakFlow's built-in performance-profiling layer: how the
framework measures where time goes in a pipeline, and how those measurements are
surfaced. Written to be read both by contributors and by stakeholders who need to
understand how LeakFlow is observed and tuned.

## 1. Goal and scope

LeakFlow runs side-channel experiments as a graph of **elements** connected by
**pads**, carrying **buffers** (see `architecture.md` and
`dataflow_sync_model.md`). A natural performance question follows directly from
that model:

> Which element — and which operation inside an element — is the bottleneck?

This layer answers that question with **domain-aware timing**: it measures time
in terms of the pipeline's own vocabulary (elements and named operations), not
just C++ functions. It is the primary performance tool. External profilers
(`perf`, `callgrind`/`kcachegrind`) remain complementary for deep dives into a
single element (see §9), but only LeakFlow knows what an "element" is.

In scope:

- Per-element execution time (free, automatic; the executor measures it).
- Opt-in timing of internal operations inside an element ("op scopes").
- A summary table (`--print-profile`) and a Chrome trace export
  (`--profile-file`).

Out of scope (deferred, see §10): per-pad timing, `Queue`-wait/backpressure
accounting, GPU/CUDA kernel timing, and a live timing overlay in `--graph`.

## 2. The one architectural decision

Timing **reuses the existing telemetry plumbing but has its own gate.**

LeakFlow already had a telemetry subsystem (`telemetry.hpp`): size gauges and
counters (e.g. `Queue` depth), declared on descriptors, surfaced through the
observer stream and the `--graph` panel, gated by a `RuntimeTelemetrySwitch`.
Timing is **just another kind of telemetry** — a runtime measurement, never a
replay/invalidation trigger — so it rides all of that plumbing. Building a second,
parallel "profiler" subsystem would have duplicated the switch, the snapshot
path, the descriptor exposure, and the `leakflow-ls` integration for no benefit.

But timing must **not** share the on/off switch with size telemetry, for two
reasons:

| | Size telemetry (e.g. queue depth) | Timing (profiling) |
|---|---|---|
| Cost | one atomic store — free | two clock reads per call; per-event spans allocate |
| Observer effect | none | profiling perturbs the very timings it measures |
| Default with `--graph` | on (cheap to leave on) | off (don't tax every interactive run) |

So there are **two gates over one data model**:

- `TelemetryKind::Size` → gated by the **runtime telemetry switch**
  (`--telemetry`).
- `TelemetryKind::Duration` → gated by the **profiling switch**
  (`--print-profile` / `--profile-file`).

The phrase to remember: **share the plumbing, split the gate.**

## 3. Building blocks (`leakflow_core`)

All timing primitives are domain-free, so they live in `leakflow_core` alongside
the telemetry they extend. `leakflow_core` still knows nothing about Torch, AES,
plotting, or the GUI.

### `RuntimeTelemetryDurationStat` (`telemetry.hpp`)

A lock-free timing accumulator: `count`, `total`, `min`, `max` in nanoseconds.
Mirrors the existing `RuntimeTelemetrySizePeak` style (atomics, CAS loop for
min/max), so it is safe to `observe()` from one thread while a reader reads the
aggregate. Bound to the **profiling** switch, it becomes a no-op when profiling
is off. It reports a `TelemetryDurationReport` (the row the table renders).

### `RuntimeTelemetryScopedTimer` (`telemetry_trace.hpp`)

An RAII scope: on destruction it records elapsed wall time into a duration stat
and, when a trace sink is set, also appends a span. It is **inactive** when both
the stat and sink pointers are null, so the cost when profiling is off is a
single null check. This one type is reused by both the executor (per-element
timing) and elements (op scopes).

### `TelemetryTraceSink` (`telemetry_trace.hpp` / `.cpp`)

A thread-safe recorder of per-event spans. Only created when the user asks for a
trace file — this is the heavy, per-event path. It exports **Chrome Trace Event
JSON** (`to_chrome_json()`), the format read by `chrome://tracing` and Perfetto.
LeakFlow therefore needs no flame-graph GUI of its own.

Two export details matter for correctness in Perfetto:

- **Fractional microseconds.** Timestamps and durations are recorded in
  nanoseconds and emitted as fractional microseconds. Integer-microsecond
  truncation collapses sub-microsecond and nested scopes onto one timestamp, and
  Perfetto then drops them as overlapping complete events.
- **One track per element, not per OS thread.** Each element gets its own trace
  track (the OS thread is kept in span `args`). An element processes one buffer at
  a time, so its complete (`X`) slices are always sequential and its op scopes nest
  strictly within them — there are never overlapping slices on a track, which is
  what Perfetto rejects. This holds across threaded live segments and `--graph`
  re-runs, and reads as a natural per-element timeline.

## 4. Where the measurements come from

### Per-element time — automatic, no element code

The executor (`Pipeline`) is already the single driver of every element; it is
the one place that calls `process_pads(...)`, and it is the single writer of the
buffer vector clock. It is therefore the natural single instrumentation point.
`Pipeline::timed_process_pads()` wraps that call in a scoped timer feeding the
element's built-in **`process`** duration channel, on both the offline and
threaded executor paths. When profiling is off it forwards directly — zero
overhead.

The `process` channel is declared for every element by
`with_common_element_properties(...)` (the same descriptor helper that publishes
the common `name` property), so `leakflow-ls` advertises it on every element and
the matching stat is pre-created.

### Internal op time — opt-in, two lines

An element declares a duration telemetry spec and opens a scope around the work
it wants to time:

```cpp
// descriptor():
.telemetry_specs = { make_duration_telemetry_spec("leakage_compute", "...") },

// process_inputs():
torch::Tensor leakage;
{
    auto scope = profile_scope("leakage_compute");   // nests under "process"
    leakage = compute_leakage(...);
}
```

Shipped examples: `AesLeakage.leakage_compute`, and `CpaAttack`'s
`prepare`/`correlation`/`score` (which reveal that its cost is ~⅔ the correlation
matmul and ~¼ the score sort). `profile_scope` is a no-op when profiling is off or
the channel was not declared, so it is safe to leave in `process()` unconditionally.

### Threading

Duration stats use atomics, and the trace sink is mutex-guarded, so both are safe
across the threaded live segments (one `std::jthread` per segment, cut at every
`Queue`). Each element runs on a single thread, and the duration-stat map is
**fixed before the run** (built in the constructor + `configure_from_descriptor`,
never mutated on a segment thread), so op scopes only ever *look up* a stat — no
concurrent map insertion. Because segments overlap in time, the table reports
**per-element wall time** and must not be naively summed across threads to get a
single "total run time"; this is also why `callgrind` (which serializes) would
mislead on live pipelines.

## 5. Collection vs. presentation

A useful mental model: **"profile" is not a separate thing being collected — it
is a view of timing telemetry.**

```text
COLLECTION (runtime, core)              PRESENTATION (frontend, CLI)
───────────────────────────            ──────────────────────────────
profiling switch ON                    how do I want to see it?
  └─ duration stats (per element + op) ─┬─→ --print-profile : table to stdout
                                        └─→ --profile-file  : Chrome trace JSON
```

The runtime only ever *collects* duration aggregates and (optionally) spans. The
CLI frontend decides whether to render a table, write a trace, or neither.

## 6. CLI surface

| Flag | Gate it enables | Output | Cost |
|---|---|---|---|
| `--telemetry` (pre-existing) | size/monitoring telemetry | live `--graph` panel | ~free |
| `--print-profile` | profiling (duration) | per-element timing table at exit | clock reads |
| `--profile-file PATH` | profiling (duration) + per-event recorder | Chrome trace JSON | per-event |

Both profiling flags are frontend conveniences that turn the **profiling** gate
on; neither forces size telemetry on. They work in headless, `--graph`, and live
plot runs.

Example table (`--print-profile`), sorted bottleneck-first:

```text
LeakFlow profile (wall time)
ELEMENT             OP                   CALLS    TOTAL ms      AVG us      MAX us
aesleakage0         leakage_compute          1      11.842     11842.0     11842.0
aesleakage0         process                  1      12.310     12310.0     12310.0
torchfilesrc0       process                  1       3.108      3108.0      3108.0
```

Example trace export:

```bash
leakflow run --profile-file /tmp/run.json 'TorchFileSrc(...) ! AesLeakage ! FakeSink'
# then open /tmp/run.json in chrome://tracing or https://ui.perfetto.dev
```

## 6a. Live `--graph` timing overlay

Under `--graph` (with profiling on, e.g. `--graph --print-profile`), per-element
timing is shown **live** in the graph's Telemetry panel:

- `Element::telemetry_snapshot()` surfaces each recorded duration channel as a
  total-milliseconds value (`kind = Duration`, unit `ms`) — the built-in `process`
  timer plus any op scopes. The stats map is fixed before the run and the counters
  are atomics, so the publisher thread reads them without locking.
- The threaded runtime publisher (the ~33 ms `--graph` sampler) publishes every
  element's duration channels as `TelemetryChanged` events, not just `Queue` size
  telemetry. The graph merges them into its per-element display.
- The graph node panel renders timing in a **separate "Profile" section**, distinct
  from the size-counter "Telemetry" section. The split is driven by `TelemetryKind`,
  which is carried on `PipelineTelemetrySnapshot` / `PipelineTelemetryChangeObservation`
  through to the UI.
- Gated by the profiling switch independently of size telemetry, so
  `--graph --no-telemetry --print-profile` still streams timing (size gauges off).
- Headless runs attach no observer, so the publisher is inert there; the table and
  trace remain the headless outputs. Verified by a headless test that attaches an
  observer to a threaded run and asserts duration `TelemetryChanged` events arrive
  (`tests/core/threaded_runner_test.cpp`).

## 7. Inspection (`leakflow-ls`)

`leakflow-ls` never runs a pipeline, so it shows the timing **capabilities**
(schema), never measured values. Each declared duration channel is listed under
the element's telemetry with a `profiling` flag (size channels show
`monitoring`). The numbers only come from an actual `leakflow run`.

## 8. Testing policy

Timing is non-deterministic, so tests assert **structure, never durations**:
that a stat records the expected `count`, that a scoped timer produces one span,
that the trace JSON is well-formed, that an element's op scope and the executor's
`process` channel both appear in `duration_reports()` after a run, and that
profiling is a no-op when disabled. See `tests/core/profiling_test.cpp`.

## 9. Relationship to external tools

| Tool | Use it for | Caveat |
|---|---|---|
| **LeakFlow profiling** | "which element/op is the bottleneck", flame graph via trace export | coarse inside an element |
| `perf record -g` | whole-program CPU hot spots, real threads, ~no slowdown | sampling; needs frame pointers/DWARF |
| `callgrind` + `kcachegrind` | exact call graph of one hot function flagged by the table | 20–50× slower, serializes threads, misleads on Torch/live timing |

Build the in-tree tool first; reach for `perf`/`callgrind` to drill into a single
element it points at.

## 9a. Queue wait-time profiling

A thread boundary (`Queue`) is not executed by the engine — the executor calls its
`boundary_push` / `boundary_pull` handoff, not `process_pads` — so a `process`
timer on it is meaningless. The meaningful Queue timing is **wait time**, which
`Queue` records as two duration channels:

- **`backpressure`** — time the producer blocked on a full queue (downstream can't
  keep up).
- **`starvation`** — time the consumer blocked on an empty queue (upstream can't
  keep up).

These are the clearest "is this element compute-bound or just stalling?" signal in
a live pipeline. They surface like any other duration channel (Profile section,
`--print-profile` table, trace, `leakflow-ls`).

Note: under a drop policy (DropOldest/DropNewest) push never blocks, so
`backpressure ≈ 0` even when the queue is overwhelmed — there the `dropped` size
counter is the overwhelm signal.

## 10. Deferred work

- **Per-pad timing** (per-link granularity below the element).
- **GPU/CUDA timing** — Torch ops are async, so wall-clock around a kernel launch
  times the launch, not the GPU work; doing this honestly needs `cudaEvent` / the
  Torch profiler. Today's tensors are CPU, so wall-clock is honest.
