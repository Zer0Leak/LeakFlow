# Core Module Context

Use this for work touching `leakflow_core`, `leakflow_render`, or the core
execution model.

## Files

Public headers:

- `include/leakflow/core`
- `include/leakflow/render`

Sources:

- `src/core`
- `src/render`

Tests:

- `tests/core`
- `tests/render`

## Targets

- `leakflow_core`
- `leakflow_render`

## Current Core API

- `Caps`: lightweight type plus deterministic string parameters.
- `Buffer`: caps, metadata, optional shared payload, vector-clock provenance, and units.
- `Payload`: polymorphic data body with `type_name()`, a required non-empty
  logical `layout()`, and optional `describe(...)`.
- `Units` (`units.hpp`): which unit each row of the payload's leading axis is (AES
  byte, Kyber coefficient) — `none` / a `range` / an explicit set, with one grammar
  (`none` / `[0]` / `[0:16]` / `[0,1,4:7]`, upper bound exclusive) for the property
  input and the display. A typed, immutable value on `Buffer::units()`; a per-unit
  fusion aligns its inputs on it before comparing.
- `Element`: lifecycle, pads, properties, and `process(optional<Buffer>)`.
- `Element` always has a writable string `name` property synchronized with
  `Element::name()`.
- `Pad`: named input/output declaration with caps.
- `Pipeline`: DAG execution with explicit links, `Tee` fan-out, and multi-input
  joins; offline is synchronous one-shot, live is a threaded pump loop (segments
  cut at every `Queue`, `run_threaded`); plus element lookup by instance name or
  descriptor type.
- `PropertyValue` / `PropertySpec`: inspectable typed configuration.
- `PropertyEffect`: per-property change behavior for UI-only, sink-display,
  metadata-output, payload-output, caps-output, and lifecycle changes, with an
  invalidation scope and optional affected output pads.
- `ElementDescriptor` / `PluginDescriptor`: linked descriptor metadata.
- `with_common_element_properties(...)`: descriptor helper that publishes the
  common `name` property with a default such as `tee0` or `filesrc0`.
- `ElementMetadataDescriptor`: typed inspect metadata for keys an element stamps
  or suggests users may stamp, with optional pad targets.
- `PipelineObserver` / `PipelineEvent`: copied topology and routed-buffer
  observation snapshots for graph/inspection UIs, plus copied
  `CommandAccepted/Rejected/Applied` control events.
- `PipelineSession`: control/session layer owning a `Pipeline`, command
  queue, cache, downstream rerun, monotonic `generation` counter, and state
  machine.
- `Pipeline` (renamed from `LinearPipeline` in Phase 27) primitives: `start_all`
  / `run_sweep` / `rerun_from` / `stop_all`, plus a per-input-pad /
  per-output-pad buffer cache. Elements produce per-pad `ElementOutputs` via
  `process_pads(...)`; `Tee` owns fan-out (no implicit engine broadcast).
- `Buffer::provenance()` + `provenance.hpp`: vector-clock buffer provenance
  (Phase 27) — slot per producer (`ElementDescriptor::provenance_slots`), engine
  increment on emission, `merge_provenance` fold-match at joins,
  `provenance_generation` for UI. Replaces the removed `Buffer::epoch()`. See
  `docs/design/dataflow_sync_model.md`.
- `Buffer::set_payload(...)` owns the reserved `payload.layout` metadata key:
  setting a payload replaces it with `Payload::layout()`, an empty layout is
  rejected, and clearing the payload erases the key.
- `Element::can_replay()`: replay-safety signal (default true).
- `Element::is_live()` / `at_end_of_stream()` / `set_stop_token()`: liveness +
  3-state stream result + cooperative stop.
- `Element::cooperative_checkpoint()`: session-injected pause/stop checkpoint for
  long synchronous `process()` loops. The executor discards any output returned
  after Stop before observing, caching, or routing it.
- `ProgressStatus` (`Active` / `Completed` / `Cancelled`) travels with element
  progress observations; terminal statuses are normalized to 100% and bypass
  progress throttling. The pipeline-owned progress sink tracks each element's
  latest status: requested Stop closes any still-`Active` report once as
  `Cancelled`, preserves explicit terminal reports, and detaches every progress
  sink before the authoritative `Stopped` event so no later report can escape.
- `BufferQueue` (`buffer_queue.hpp`): thread-safe bounded FIFO with
  Block/DropOldest/DropNewest; `pipeline_segments.hpp` decomposition and
  `Pipeline::run_threaded` (one `std::jthread` per segment).
- `QueueEpochPolicy`: documented enum; the drain/flush generation-boundary knob is
  an optional policy (the vector clock matches by config-independent ancestor
  identity), not a correctness requirement.
- `DescriptorRegistry`: linked/static descriptor registry for current plugin
  catalogs; not dynamic plugin loading.
- `ElementFactoryRegistry`: explicit linked/static factory registry that pairs
  plugin descriptors with element construction callbacks.
- `SummaryDocument`: structured summary data independent of rendering.
- Profiling primitives (`telemetry.hpp`, `telemetry_trace.hpp`): `TelemetryKind`
  (`Size`/`Duration`), `RuntimeTelemetryDurationStat`,
  `RuntimeTelemetryScopedTimer`, `TelemetryTraceSink` (Chrome trace JSON). The
  executor times every element's `process` step (`Pipeline::timed_process_pads`);
  elements add opt-in op scopes via `Element::profile_scope`. Gated by a profiling
  switch separate from size telemetry (profiling defaults off);
  `Pipeline::set_profiling_enabled` / `set_trace_sink` mirror to elements.
  `Element::duration_reports()` feeds the `--print-profile` table;
  `Element::telemetry_snapshot()` surfaces duration channels (ms) and the threaded
  runtime publisher streams them as `TelemetryChanged` events for the live
  `--graph` timing overlay. Design: `docs/design/profiling.md`.

## Descriptor Ownership

Element-specific descriptor details, including pads, properties, metadata set by
the element, suggested user metadata, and metadata pad targets, live with the
element implementation. Plugin descriptor catalogs should assemble plugin-level
metadata and call `ElementType::descriptor()`; they should not duplicate element
configuration.

Common element properties are not hand-written in every element descriptor.
Use the core descriptor helper through plugin catalogs or descriptor registry
registration so `name` appears consistently before element-specific properties.

## Contracts

Core must not depend on Torch, NumPy, AES, Kyber, GUI, YAML, plotting, hardware,
or dynamic plugin loading.

Every concrete payload declares its logical layout. Dense axes are
slash-separated (`axis_0/axis_1`), a scalar is `scalar`, and structured payloads
use semicolon-separated named members (`name=axes;other=scalar`). The layout is
not wrapped in the payload type name. `Buffer::set_payload(...)` publishes this
value as `payload.layout`, making the payload implementation the source of truth.

Offline execution is synchronous; live execution is threaded (segments cut at
every `Queue`). Elements stay synchronous and thread-unaware — only the engine and
`Queue`/`BufferQueue` touch threading.

Element name uniqueness is enforced by `Pipeline::add(...)`. Builders and
CLIs may generate names such as `summary0` and `summary1`, but the pipeline is
the authoritative registry once elements are added.

Once an element is accepted by `Pipeline::add(...)`, the pipeline owns the
name map and locks the element's `name` property against direct mutation.
`Pipeline::find_element(...)`, `Pipeline::element(...)`, and
`Pipeline::elements_by_type(...)` are the public lookup APIs.

Factory registration is explicit at plugin-catalog level. Do not add static
self-registration in element translation units.

Multi-input joins and the threaded live executor are implemented. Do not add full
caps negotiation, wildcard caps, or dynamic pads without an explicit phase.

Pipeline observation is diagnostic. Core may emit copied topology/property/pad
snapshots and copied routed-buffer summaries, but it must not expose mutable
element handles, mutable buffers, payload pointers, raw trace values, keys, or
plaintext contents to UI observers.

Property-change observations carry copied element identity, property name,
stringified previous/current values, value type, and `PropertyEffect` metadata.
They are diagnostic/control-plane events; partial downstream rerun semantics
remain future work.

`PipelineSession` (Phase 25) is the control/session layer above/around
`Pipeline`, in `leakflow_core` (no UI concepts). Authoritative design:
`docs/design/pipeline_controller.md`. It:

- owns a `Pipeline` (move-in) and the lifecycle (Option A: `start_all` once
  → `run_sweep`/`rerun_from` many → `stop_all` at teardown),
- accepts validated `SetProperty` commands through a thread-safe queue
  (last-wins coalescing per `(element, property)`),
- applies commands at safe points (only the worker thread mutates elements),
- uses `PropertyEffect` to decide UI-only / sink-display / metadata-output /
  payload-output / caps-output / lifecycle behavior,
- caches latest input buffers by input pad and output buffers by output pad
  (global caching toggle, default on),
- reruns only downstream links from affected output pads
  (`Pipeline::rerun_from`), escalating to a full restart when the replay-set
  contains a `can_replay() == false` element,
- revalidates downstream links before caps-output reruns (transactional reject),
- bumps a session-monotonic epoch on accepted dataflow commands,
- catches rerun exceptions without tearing down the session,
- owns a `Stopped/Started/Running` state machine and session controls (restart,
  re-run-from-sources, caching toggle),
- emits copied `CommandAccepted/Rejected/Applied` observations.

The executor is a single engine: `run()` is sugar over
`start_all`/`run_sweep`/`stop_all`; `rerun_from` is the same step engine seeded
from cache. The executor is the single writer of `Buffer::provenance()` (the
vector clock); joins fold-match it, so partial rerun is verified, not assumed.
`Element::can_replay()` (default true, mirrored to the descriptor) signals replay
safety. The live machinery is **implemented**: the unified `run()` pump loop,
threaded segments + `BufferQueue`, the aggregator fold-match, cooperative
`std::stop_token` cancel, and the `Stopped/Running/Paused/Idle` player state
machine with safe-point forward-apply. `QueueEpochPolicy` drain/flush enforcement
and `preroll` are the only remaining seams. See
`docs/design/dataflow_sync_model.md` §12–13.

Metadata-output changes are dataflow changes: they create a new `Buffer` envelope
with updated metadata (shared payload policy) rather than bypassing downstream
elements.

Observer callbacks must not be part of experiment correctness. A broken graph
consumer should not break pipeline execution.

Caps compatibility is intentionally small and deterministic:

- exact source/sink caps type match is accepted,
- concrete source caps can link to a generic `leakflow/buffer` sink,
- a generic source pad on a generic forwarding element may resolve to the
  single upstream concrete caps already linked into that element,
- if both endpoints declare the same caps parameter key, the values must match,
- missing caps parameters mean unspecified, not negotiated.

This is not full graph-wide caps negotiation. A generic source with no
resolvable upstream concrete caps still cannot link to a concrete sink.

## Common Tests

Run focused tests after core changes:

```bash
ctest --test-dir build -R 'leakflow_(caps|buffer|payload|property|element|element_factory_registry|pad|linear_pipeline|pipeline_observer|descriptor|render)'
```

For broad validation, run full CTest.
