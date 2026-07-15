# Architecture Contracts

These are the high-signal contracts to preserve during implementation.

The source design document is `docs/design/architecture.md`. This file is the
compact context version for day-to-day development.

## Core Boundary

`leakflow_core` may know:

- `Caps`
- metadata
- `Payload`
- `Buffer`
- `Element`
- `Pad`
- links
- minimal synchronous execution
- property and descriptor models
- structured summary documents

`leakflow_core` must not know:

- Torch / LibTorch
- NumPy / cnpy++
- HDF5
- AES
- Kyber / ML-KEM
- CUDA implementation details
- YAML or config formats
- GUI concepts
- plotting
- hardware capture
- dynamic plugin loading

## Buffer And Payload

The pipeline is `Buffer`-centric:

```text
Buffer = Caps + metadata + zero-or-one Payload + vector-clock provenance + units
```

Every routed buffer carries a **vector clock** (`Buffer::provenance()`, Phase 27):
a dense per-element production-count vector the executor stamps as it routes.
Joins fold-and-match these clocks (`merge_provenance`) so branches that reconverge
are verified to share a generation; a conflict is a desync. This replaced the
removed `Buffer::epoch()`. See `docs/design/dataflow_sync_model.md`.

A buffer also carries **units** (`Buffer::units()`, a `Units`): which unit each row
of the payload's leading axis is (AES byte, Kyber coefficient). It is a typed,
immutable value on the envelope — not metadata, which the Analyze profile would drop
— set by the producing element next to the payload and copied per branch on a `Tee`.
`Units` is `none` / a `range` / an explicit set, with one grammar (`none` / `[0]` /
`[0:16]` / `[0,1,4:7]`, upper bound exclusive) shared by the property input and the
display. A per-unit fusion (e.g. `ClusteringStats`) aligns its inputs on units before
comparing: disjoint units are an error, a partial overlap warns and scores the shared
units. See `docs/design/metadata_klass_taxonomy.md`.

A payload may internally be a bundle or batch.

Queues store `Buffer` objects, not payloads inside one buffer.

`set_payload(nullptr)` clears the payload.

Every concrete `Payload` implements `layout()` and returns a non-empty logical
layout. Dense axes use slash-separated names (`axis_0/axis_1`), scalars use
`scalar`, and structured payloads use semicolon-separated named members
(`name=axes;other=scalar`); the payload type is not wrapped around the layout.
`Buffer::set_payload(...)` is the source-of-truth boundary for the reserved
`payload.layout` metadata key: setting a payload replaces the key with
`Payload::layout()`, an empty layout is rejected, and clearing the payload erases
the key.

Use `payload_as<T>()` for read-only typed access.

Use `mutable_payload_if_unique<T>()` before in-place mutation. If the payload is
shared, a mutating element must create and set a replacement payload.

## Buffer Persistence

A whole `Buffer` (caps + metadata + payload) round-trips through
`BufferFileSink`/`BufferFileSrc` (in `leakflow_plugins_extras`) to a single HDF5
file — the `leakflow.buffer` schema, following the trace tensor-dataset
conventions: the envelope as attributes (root `caps.type`/`payload.type`, `/caps`
and `/metadata` attribute groups) and the payload body as native datasets under
`/payload`. Provenance (the vector clock) is **not** persisted; a reloaded buffer is
a fresh source production the executor re-stamps.

Serialization is layered so no layer learns a foreign domain:

- `PayloadCodecRegistry` (in `leakflow_core`) maps a payload `type_name()` to a
  `PayloadCodec { save, load }`. Core holds only the callbacks and **forward-declares**
  the archive types, so it stays free of Torch and HDF5.
- `BufferArchiveWriter`/`BufferArchiveReader` (in `leakflow_base`) is the
  storage-neutral, Torch-aware, HDF5-agnostic seam a codec writes through
  (`write_tensor`/`write_int`/`write_string` + read mirror). Payload codecs in base
  and crypto serialize their payloads as named tensors/scalars against it, never
  seeing the storage backend.
- `Hdf5BufferArchiveWriter`/`Reader` (in `leakflow_extras`) is the concrete HDF5
  backend. A future Zarr backend is another implementation of the same interface —
  **no codec changes**.

The elements live in `leakflow_plugins_extras` (not core) because only that layer
links both Torch (`leakflow_base`) and the HDF5 storage backend. torch-pickle is not
used: every payload is stored as native, inspectable HDF5 arrays.

## Caps

`Caps` is currently a lightweight descriptor:

```text
type + deterministic string parameters
```

There is no full MIME parsing, wildcard matching, or full caps negotiation yet.

Generic sink pads declared as `leakflow/buffer` may accept concrete buffer caps
such as `leakflow/numpy-array`.

Generic source pads declared as `leakflow/buffer` may resolve to concrete caps
when the element has one already-linked generic input carrying upstream concrete
caps. This supports forwarding elements such as `Summary`, `Queue`, and `Tee`.
A generic source with no resolvable upstream concrete caps still does not
satisfy a concrete sink.

When both linked pads declare the same caps parameter key, the values must
match. Missing parameters remain unspecified. This is not full negotiation.

The shared numeric caps vocabulary currently uses string parameters such as:

- `dtype`,
- `device`,
- `rank`,
- `shape`.

## Metadata Forwarding And Klass

Source of truth: `docs/design/metadata_klass_taxonomy.md`.

Every stamped metadata key carries its group as the leading segment;
`metadata_group(key)` is a direct lookup of that segment:

- `capture.*` — durable acquisition/dataset/countermeasure facts
  (`capture.source`, `capture.sample_rate_hz`, `capture.dataset.name`,
  `capture.countermeasure.*`).
- `origin.*` — per-input and storage provenance (`origin.file.*`,
  `origin.storage.*`, `origin.role`, and fused `origin.<pad>.*`).
- `payload.*` — producer assertions about the current bytes (`payload.leakage.*`,
  `payload.crypto.*`, `payload.poi.*`, `payload.conversion.*`, ...). Unprefixed or
  unknown keys also resolve here, so `leakflow_core` stays domain-free.
- `routing.*` — `routing.element` and `routing.branch.*`; never forwarded.

Element `klass` is `<Profile>/<Family>[/<Role>[/<Variant>...]]`. The leading
token is the forwarding profile, mapped by `profile_for_klass(klass)`: `Source`,
`Sink`, `PassThrough`, `Convert` (Reframe), `Analyze`. Later tokens are the
human-facing taxonomy (`Flow`, `Inspect`, `Score`, `PoI`, `Attack`,
`Evaluation`, `PlotAnnotation`, ...).

The graph palette is derived from the same hierarchy. New first-class `klass`
families/roles should update both `docs/design/metadata_klass_taxonomy.md` and
`klass_colors(...)` in `src/plot/pipeline_graph.cpp`.

New-buffer-building elements call `forward_metadata(inputs, profile, output)`
before stamping their own keys:

- PassThrough copies capture + origin + payload (drops routing).
- Reframe copies capture + origin (drops payload + routing).
- Analyze unions capture (a conflicting value across inputs is an error),
  relabels origin as `origin.<pad>.<key>`, and drops payload + routing. A
  `Reference` input pad is excluded from the capture union — a parameter carried
  from another experiment (e.g. `PoiCorrelation` applying profiling PoIs to attack
  traces), whose facts forward as provenance `origin.<pad>.*` instead of colliding
  with the output's own capture identity.

An Analyze element may additionally re-own a curated subset of payload facts it
asserts about its output (for example `PoiSelect` keeps the target model's
`leakage.*`/`crypto.*`). Pass-through elements that return the input buffer keep
the envelope, including routing, until the next reframe/analyze drops it.

## Elements And Pads

Elements declare input/output pads and properties.

Elements produce a buffer **per output pad** (`Element::process_pads(...)` →
`ElementOutputs`, Phase 27). The default routes the single `process_inputs(...)`
result to the sole output pad; multi-output elements (`Tee`) override to emit per
pad. **Fan-out is a `Tee` behavior, not an engine rule** — the executor routes
each pad's buffer to that pad's link; there is no implicit "one buffer → all
pads" broadcast. The executor is the single writer of each buffer's vector clock.

Every element has a readable/writable string `name` property. The instance name
is also the element identity used by logs, summaries, CLI references, and link
diagnostics. Descriptor catalogs and `leakflow-ls` expose the common `name`
property before element-specific properties.

Default generated names use the lower-case alphanumeric element type plus a
zero-based index, for example `tee0`, `tee1`, and `filesrc0`. Duplicate explicit
names are errors; they are not auto-renamed.

`Pipeline` owns the element-name registry for added elements. Adding an
element with a duplicate instance name is an error, and accepted elements have
their `name` property locked against direct mutation. Rename-after-add must go
through a future pipeline API if it is ever needed.

`Pipeline` exposes lookup by exact instance name and by element type. Type
lookup is based on the element descriptor type name, not on C++ RTTI.

Pad links are validated against element handles, pad names, pad directions, and
simple caps compatibility.

The executor supports linked DAG paths, `Tee` fan-out, and **multi-input joins**
(collect-then-fire, gathering one buffer per input pad and fold-matching their
vector clocks). Offline runs are synchronous/one-shot; **live runs are threaded**
(segments cut at every `Queue`, one thread per segment) via the same
`Pipeline::run()` pump loop.

**Synchronization model (finalized AND implemented; design of record
`docs/design/dataflow_sync_model.md` Sections 10–14).** The default sync at every
join is the **per-slot Barrier** (the vector-clock fold-match): common-origin
inputs are auto-synced, independent inputs fire free — **99.99% of pipelines need
no tuning**. Each slot is **one counter** (generation); there is **no global
position/version split**. Custom cross-source pairing is done **only** through an
explicit **Sync element** (`N→N`, generic, injects a common ancestor so downstream
uses the default barrier) — not via per-pad flags. Property changes are
**liveness-aware**: sources are `live` or `one-run` (default one-run, propagated
downstream by OR); a change on a live-driven element applies **forward** (next
buffer), on a one-run-driven element re-emits from cache. Threaded segments, the
real `Queue`/`BufferQueue`, the `Sync` element, and the liveness model are
**implemented** (the live phase); bundles/`Mux`/`Demux` are **dropped** (superseded
by the `Sync` element).

## Pipeline Observation

Core owns neutral observation contracts only:

- `PipelineObserver`
- `PipelineEvent`
- copied topology snapshots
- copied routed-buffer observations
- copied property-change observations

The observer API is for inspection and UI state transfer, not data transport.
It must not expose mutable `Element` handles, mutable `Buffer` handles, payload
pointers, raw trace values, keys, plaintext arrays, or other secret material.

Topology snapshots may include:

- element type/name/kclass,
- current copied property values,
- property effect declarations,
- declared pads and pad templates,
- declared/resolved link caps.

Routed-buffer observations may include:

- source and sink endpoints,
- buffer caps,
- metadata,
- payload presence and payload type,
- sequence counters.

Property-change observations may include:

- element type/name/kclass,
- property name,
- stringified previous/current values,
- value type,
- `PropertyEffect` metadata.

The UI must consume snapshots/events and maintain its own display copy. It must
not query live `Element` objects every frame from another thread.

Offline execution is synchronous; live execution is threaded (segments cut at
every `Queue`). `leakflow run --graph` runs the pipeline on a worker (and, for
live + `Queue` graphs, additional segment threads) while the main thread owns
ImGui. The worker emits observer events; the UI drains them. The pipeline worker
must not call UI code, and the UI thread must not mutate pipeline internals
(edits go through the session's safe-point control plane).

Observer callbacks are best-effort diagnostics. They must not be required for
experiment correctness, and observer failures must not break pipeline execution.

## Control Plane

Graph observation and element control are separate planes.

`PipelineGraphRuntime` stores copied display state only. `PipelineControlRuntime`
renders controls from `PropertySpec` / `PropertyValue`; since Phase 25 it is a UI
client of the session and submits commands rather than mutating elements.

`PipelineSession` (Phase 25, in `leakflow_core`) is the control/session layer.
Authoritative design: `docs/design/pipeline_controller.md`. It owns a
`Pipeline` (move-in) and:

- accepts validated `SetProperty` commands through a thread-safe queue with
  last-wins coalescing per `(element, property)`;
- applies commands only at safe points (between units of work — between sweeps
  offline, between buffers for future live mode); only the worker thread mutates
  live elements;
- uses `PropertyEffect` to choose ui-control / sink-display / metadata-output /
  payload-output / caps-output / lifecycle behavior;
- caches the latest input buffer per input pad and output buffer per output pad
  (global caching toggle, default on; off disables partial rerun → full
  re-sweep);
- reruns only downstream links from the affected output pad(s)
  (`Pipeline::rerun_from`), escalating to a full restart when the replay-set
  contains an element with `can_replay() == false`;
- validates caps-output changes against downstream links before commit
  (transactional reject);
- bumps a session-monotonic `generation` counter on accepted dataflow commands
  (Phase 27, renamed from epoch); per-buffer generation/sync is carried by the
  vector clock (`Buffer::provenance()`), which the executor stamps as the single
  writer — `Buffer::epoch()` was removed. See `docs/design/dataflow_sync_model.md`;
- catches rerun exceptions without tearing down the session, emitting
  `CommandApplied{status=failed}` and marking affected cache entries stale;
- owns a `Stopped/Running/Paused/Idle` player state machine (pause/resume wired;
  only `preroll` reserved) and session controls (restart, re-run-from-sources,
  caching toggle, pause/resume);
- injects `Element::cooperative_checkpoint()` pause/stop gates into long
  synchronous work; outputs returned after Stop are discarded before observation,
  caching, or downstream routing;
- emits copied `CommandAccepted/Rejected/Applied` events on the observer stream.

The graph runner uses a persistent worker loop. The `OneShotDrive`/`StreamingDrive`
split (Phase 25 reserved seams) was **collapsed into a single implemented `run()`
pump loop** — live is *not* a run mode; a one-run source ends after one buffer, a
live source streams (see `docs/design/dataflow_sync_model.md` §11.8, §12).
Streaming results are 3-state (`Data`/`NoData`/`EndOfStream`), and
`stop()`/SIGINT/window-close drive a cooperative `std::stop_token` for a graceful,
prompt shutdown. The session owns a `Stopped/Running/Paused/Idle` player state
machine (§13). `QueueEpochPolicy` drain/flush enforcement is left as an optional
policy (not a correctness requirement, since the vector clock matches by
config-independent ancestor identity).

Element properties are the source of truth for property-backed controls. Graph
gear controls, standalone control panels, and future element-local controls
such as a `TracePlot.trace_index` slider should use the same property/control
path when they control an element property.

`PropertyEffect` describes what a property change means:

- `ui-control` — non-dataflow control/presentation. The owning element applies it
  to its own UI/display state (e.g. in `property_changed`); the session does
  nothing (no rerun, no cache, no generation) and it is valid in any player state
  (Running / Paused / Idle / Stopped). All `TracePlot` properties (center0, color,
  layout, x_axis, title, …) are `ui-control` and self-applied to the live snapshot.
- `sink-display` — the sink must re-derive its display from its cached **input**
  buffer (output depends on buffer content + property), so the session reprocesses
  the sink from cache (no upstream rerun). This needs the input cache, so it is
  legitimately state-sensitive.
- `metadata-output` — output buffer metadata changes; downstream dataflow is
  affected.
- `payload-output` — output payload changes; downstream dataflow is affected.
- `caps-output` — output caps may change; downstream links must be revalidated.
- `lifecycle` — requires full-pipeline restart or lifecycle handling. The
  `--graph` control panel therefore makes a `lifecycle` property **editable only in
  the `Stopped` state** (read-only, marked `(stop to edit)`, while
  Running / Paused / Idle), because a lifecycle edit could not take effect until a
  restart anyway. This gate is generic (`draw_element_controls`), so it applies to
  any lifecycle property — e.g. a live source's `path` or row range, or an
  app-exposed `AppSrc` knob.

Dataflow changes are changes to caps, metadata, payload, or lifecycle. They
must not be represented as `ui-control` state. Metadata-output changes should
create a new `Buffer` envelope when rerun, even if the payload pointer is reused.

Downstream-only rerun uses cached/latest input buffers per element input pad,
cached/latest output buffers per output pad, dirty output tracking, and downstream
link walking. Live pipelines do not silently mix old and new stream states: a
mid-run property change forward-applies and joins match by config-independent
ancestor identity (the vector clock), so a `QueueEpochPolicy` drain/flush is an
optional policy rather than a correctness requirement.

## Properties

Properties are small user-controlled settings, not data transport.

Valid property value families:

- bool
- integer
- double
- string
- integer interval
- double interval
- integer list
- double list
- string list

Large experiment data belongs in payloads.

Property specs may also declare `PropertyEffect` and affected output pads. For
example, `AesLeakage.channels` is `payload-output` with downstream invalidation
on the `leakage` output pad because changing it changes the target tensor shape
and target metadata consumed downstream.

Properties live at two levels. An element **type** declares its properties in its
static `descriptor()` — this is what `leakflow-ls` lists, and it stays generic. An
element **instance** may be enriched at runtime through the public
`Element::add_property(PropertySpec)` — used by applications to attach per-instance
knobs (e.g. an `AppSrc` fed by an app; see
`docs/context/modules/plugins-base.md`). Both the copied observer topology
snapshot and the `--graph` control panel render an element's **live**
`property_specs()`, not the static descriptor, so app-added properties appear as
controls and ride the normal `PropertySpec` → session `SetProperty` (safe-point) →
`property_as` path. `leakflow-ls` intentionally lists only type-level descriptor
properties.

## Descriptors

`PluginDescriptor` describes a linked plugin family.

`ElementDescriptor` describes one element type.

Descriptors support CLI validation and `leakflow-ls` inspection. They are linked
into the current tools; dynamic plugin loading is future work.

`ElementFactoryRegistry` pairs linked descriptors with element factories. Plugin
catalogs should register descriptors and factories together through explicit
plugin-level registration functions. Avoid global static self-registration; it
makes link-time behavior and initialization order harder to reason about.

## Plugin vs Element

A plugin is a distribution/registration unit.

An element is one processing block used in a pipeline.

Example:

```text
leakflow_plugins_core
  provides FileSrc, FileSink, FakeSrc, FakeSink, Summary, Tee, Queue
```

Do not treat every element as a separate plugin.

## Summary Rendering

Structured summary data belongs in `SummaryDocument`.

Color, glyphs, and terminal style belong in `leakflow_render`.

Payloads may describe themselves into a `SummarySection`.

Automated tests should use deterministic no-color rendering unless explicitly
testing ANSI behavior.

## CLI Syntax

`leakflow run` is a small manual pipeline language, not YAML and not a full
graph language.

The parser currently validates:

- element type names,
- duplicate names,
- `name` property creation names,
- named references,
- pad references,
- properties,
- caps annotations,
- metadata annotations,
- ambiguous pad inference,
- link compatibility.

Caps annotations are parsed and validated but not yet applied to mutable pad
declarations. Metadata annotations on elements apply to all output pads.
Metadata annotations on output pads or output pad templates are stamped onto
routed buffers before delivery to the linked sink. Metadata specificity is:

```text
all output pads < pad template < exact pad
```

## Profiling And Timing Telemetry

Source of truth: `docs/design/profiling.md`.

Timing is time-flavored telemetry: it reuses the telemetry plumbing (specs,
snapshots, observer, `leakflow-ls`) but has its own gate. `TelemetryKind` marks a
spec `Size` (monitoring, gated by `--telemetry`, default on with `--graph`) or
`Duration` (profiling, gated by the profiling switch, default off).

- The executor is the single instrumentation point: it times every element's
  `process` step into the element's built-in `process` duration channel.
- Elements time internal operations with `Element::profile_scope("name")` against
  a declared `Duration` spec (a no-op when profiling is off).
- `--print-profile` renders a per-element timing table; `--profile-file` writes
  Chrome Trace Event JSON. Both are frontend output over the same collected data.
- `leakflow_core` stays domain-free: the timing primitives are generic, and
  timing is best-effort like the observer — never required for correctness, and
  tests assert structure, never durations.

## Long-Running Work: Progress and Cooperative Cancellation

Long `process()`/`process_pads()` calls must stay responsive to the session's
pause/stop control plane and should surface progress. There are two cases:

- **Element-owned loop** (the element itself iterates): call
  `report_progress(fraction, message, index, total, status)` to publish progress
  on the observer bus, and `cooperative_checkpoint()` between units of work.
  `cooperative_checkpoint()` parks while paused and returns `false` on stop; on a
  `false` return the element stops early and returns an empty result. Emit an
  explicit terminal `ProgressStatus::Completed` on success and
  `ProgressStatus::Cancelled` when a checkpoint aborts the work.

- **Library-owned loop** (the heavy loop lives in a compute/IO library the
  element calls, e.g. `leakflow_ml`, `leakflow_extras`): the library stays
  framework-agnostic (no `leakflow_core` types) and exposes optional
  **framework-agnostic callbacks** — a progress callback and/or a cheap
  checkpoint callback — that **return `bool`, where `false` means cancel**. The
  library aborts promptly on a `false` return (throwing a typed cancellation
  rather than returning partial data when a partial result would be unsafe). The
  calling element bridges those callbacks to `report_progress` and
  `cooperative_checkpoint()`, and translates a cancellation into an empty result
  plus a `Cancelled` report.

Reference implementations: `leakflow::ml::GaussianMixture::fit(x, on_progress,
on_checkpoint)` (both callbacks `bool`, `false` = cancel), bridged by the
`GaussianMixture` element; and `leakflow::extras::TensorDatasetReader::read_tensor`
(progress callback `bool`, `false` throws `TensorReadCancelled`), bridged by
`Hdf5FileSrc`/`FakeLiveHdf5Src`. A new backend on a shared library contract (e.g.
a future Zarr `TensorDatasetReader`) inherits the same cancellation contract.

Progress and cancellation are **best-effort diagnostics/control**, like the
observer and profiling: they must never be required for experiment correctness,
and `report_progress` is a cheap no-op when no sink is injected. Purely
vectorized elements whose heavy work is a single library/tensor call (e.g. the
CPA/DPA/correlation/leakage crypto elements) have no interruptible inner loop;
they rely on the executor's between-buffer safe points for pause/stop and do not
need in-call checkpoints.

## Future Plugin Boundaries

Future algorithm or UI features must stay outside core:

- Crypto/AES helpers: `leakflow_crypto`.
- Crypto/AES elements: `leakflow_plugins_crypto`.
- Generic SCA elements: `leakflow_plugins_sca`.
- Kyber elements: `leakflow_plugins_kyber`.
- Plotting runtime + views: `leakflow_plot` (domain-free).
- Plot elements: `leakflow_plugins_plot` (`TracePlot`).
- Crypto→plot bridge elements: `leakflow_plugins_crypto_plot` (`ScorePlot`,
  which reads `AttackStatsPayload` and fills a `ScoreView`). A plot element that
  needs domain payloads goes in a bridge plugin, never in `leakflow_plot`.
- GUI: a separate app or plugin layer.

Do not pull future plugin dependencies into core.

A new plot **type** is a new `leakflow::plot::PlotView` (its own copied data, UI
state, rendering, and lock) registered with `PlotRuntime::add_view`, not a new
branch inside the runtime. `PlotRuntime` is to `PlotView` what `Pipeline` is to
`Element`: it owns no plot data itself and draws/clears the registered views
without knowing their kind, so plot growth stays out of the shared runtime. Even
the built-in trace plot is a view (`TraceView`, created and registered at
construction); there is no special-cased trace path in the runtime or draw loop.
`leakflow_plot` holds only domain-free view **display** data (e.g. `ScoreView` =
panels/series/points); the crypto→generic translation lives in the bridge
element. Design: `docs/design/plotting.md` (Plot View Architecture).
