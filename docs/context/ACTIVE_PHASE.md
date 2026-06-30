# Active Phase Brief

This file is the compact phase/task brief for Codex.

## Current Default

**No active phase.** The live-streaming phase and the AES CPA/report slice are
implemented. Pick the next phase explicitly before starting work (candidates:
dedicated CPA score/rank/heatmap plots, overlay/correlation plot polish, or
Kyber / ML-KEM hypotheses).

The **execution timing telemetry / profiling phase** is implemented (the
"Execution Timing Telemetry and Trace Export" phase in `ROADMAP.md`): per-element
`process` timing, opt-in `Element::profile_scope` op scopes, `--print-profile`
table, `--profile-file` Chrome trace, and a live `--graph` timing overlay
(per-element duration telemetry published live to the graph panel), gated
separately from size telemetry (profiling defaults off). Design of record:
`docs/design/profiling.md`. `Queue` profiles wait time (`backpressure` /
`starvation`). Deferred seams: per-pad timing, GPU/CUDA timing.

Recently completed, in order:

- **Phase 25** (`PipelineController`/session layer) — design in
  `docs/design/pipeline_controller.md` (`D1`..`D12`).
- **Phase 26** (AES PoI pipeline correctness validation) —
  `tests/plugins/crypto/aes_poi_correctness_test.cpp` proves correct
  leakage/PoI/annotation values over `key_01`/`key_02`.
- **Phase 27** (DAG executor, per-pad outputs, vector clock) —
  per-pad `ElementOutputs`; `Tee` explicit fan-out; the **vector clock**
  (`provenance_slots`, dense `Buffer::provenance`, conflict-detecting fold-match);
  `Buffer::epoch()` removed (session `generation` counter); `LinearPipeline` →
  `Pipeline`. Bundles/`Mux`/`Demux` (old 27.4/27.5) were **not built** — the
  vector clock subsumes their offline role.
- **Live phase** (the largest) — implemented and green; offline unchanged.
  Authoritative design: `docs/design/dataflow_sync_model.md` §10–14.

Generic `Convert`, the conversion registry, dedicated CPA score/rank/heatmap
plots, and Kyber hypotheses remain deferred.

## Phase 26 Brief (DONE)

Implemented. Numeric correctness is asserted in
`tests/plugins/crypto/aes_poi_correctness_test.cpp` for `key_01`/`key_02`:
`AesLeakage` HW(m)/HW(y) vs values computed directly from the fixture bytes,
`PearsonPoiFinder` PoI landing on the strongest-correlation sample with an
independently recomputed correlation clearing a sane threshold, and
`CorrelationPoiToPlotAnnotations` sample indexes/values/precision formatting
matching the PoI output. No dependency on the local `traces/` tree.

## Phase 27 Brief (DONE)

Authoritative design: `docs/design/dataflow_sync_model.md`. Offline / one-shot;
live and Queue (and bundles/Mux/Demux) deferred to the next phase.

```text
Phase 27 — DAG executor, per-pad outputs, and vector-clock buffer provenance
(offline). Buffer::epoch() removed. DONE; 120/120 green.

Delivered:
  27.1 per-pad ElementOutputs; uniform per-pad routing; Tee explicit fan-out;
       implicit "one buffer -> all pads" broadcast removed. Topological validity
       and acyclicity are structural properties of link()'s low->high index rule
       (no separate Kahn pass needed).
  27.2 Vector clock: ElementDescriptor::provenance_slots; Pipeline slot allocation
       (monotonic counter, 0 reserved, wrap max->1; Tee/sinks = 0 slots);
       Buffer::provenance() dense uint32 clock; engine increments the producer's
       slot on emission; merge_provenance() fold-and-detect match at joins.
       Tests: tests/core/vector_clock_test.cpp (provenance fns incl. N=3 conflict,
       join provenance, fan-out rejoin match, partial-rerun supersede).
  27.3 Buffer::epoch() removed. PipelineSession epoch_ -> generation_ /
       generation(); observer + plot use a buffer "generation" derived from the
       clock (provenance_generation). Partial-rerun orchestration unchanged
       (cache + replay-set + downstream walk); consistency now verified by the
       fold-match. QueueEpochPolicy kept as the reserved live-phase enum.
  27.6 class rename LinearPipeline -> Pipeline; files pipeline.{hpp,cpp}. Test
       file names (linear_pipeline_*_test.cpp) kept.

NOT built (moved to live phase; superseded by the vector clock for offline):
  - bundles, BundlePayload interface, generic Mux/Demux (old 27.4);
  - AesLeakage single-bundle pad (old 27.5) -- its separate trace/plaintext/key
    pads are now safe via the fold-match.

Build/Test:
  CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
  cmake --build build -j && ctest --test-dir build --output-on-failure
```

## Live phase — DONE

Implemented and green, offline unchanged. Implementation map + tests:
`docs/design/dataflow_sync_model.md` §12.

- **Unified `run()` pump loop** (`Pipeline::run()`): auto-detects a live source and
  pumps until every live source is at end-of-stream; offline = one sweep. No
  `run_live()`. `Element::is_live()` (descriptor `live_source`) + `at_end_of_stream()`.
- **`FakeLiveSrc`** (`leakflow_plugins_base`): reads a Torch `.pt`, streams one
  `Buffer` per axis-0 row then EOS; `trace_rate` pacing; declares itself live.
  Test `tests/plugins/base/fake_live_src_test.cpp`.
- **Liveness propagation** `Pipeline::is_live_driven()` (OR-reachability from live
  sources); **liveness-aware property change** in `PipelineSession::apply_command`
  (forward-apply on live-driven, cache re-emit on one-run-driven).
- **Segment decomposition + threaded runner** (`pipeline_segments`,
  `Pipeline::run_threaded` / `run_source_segment` / `run_consumer_segment`): cut at
  every `Queue`, one `std::jthread` per connected component. Tests
  `pipeline_segments_test`, `threaded_runner_test`.
- **`Queue` = `BufferQueue`** (`buffer_queue`): bounded FIFO; Block backpressure vs
  DropOldest/DropNewest; thread + generation boundary. Test `buffer_queue_test`.
- **Aggregator**: Barrier fold-match across queue heads, realign-on-drop, Held
  (auto from liveness), Latest. Tests `threaded_runner_test`, `aggregator_latest_test`.
- **`Sync` element** (`leakflow_plugins_core`, `N→N`): claims a slot, stamps all
  aligned outputs alike (common-ancestor injection); `policy` =
  barrier/zip/all-required-once/held/latest (restart-scoped). Tests
  `tests/plugins/core/sync_test.cpp`, `sync_threaded_test.cpp`.
- **Cooperative stop** (`std::stop_token` through element→pipeline→session; CLI
  SIGINT bridge; `--graph` window close) and the **player state machine**
  (`Stopped/Running/Paused/Idle`, pause/resume, safe-point forward-apply). Tests
  `threaded_safe_point_test`, `threaded_pause_test`.

Only `preroll` and the optional `QueueEpochPolicy` drain/flush enforcement remain
unbuilt seams.

## (original brief) Live streaming + Sync element

> **Status: DELIVERED.** This is the original planning brief, kept for history;
> all work items below are implemented (see "Live phase — DONE" above and
> `docs/design/dataflow_sync_model.md` §12).

Design of record: `docs/design/dataflow_sync_model.md` Sections 10–12 (finalized
synchronization & liveness model). **Implementation reference + validation:**
`docs/design/dataflow_sync_walkthroughs.md` — ten worked pipelines showing threads,
join behavior, and the vector-clock indexes step by step (offline AES diamond,
independent sources + Sync element, live streaming, live + Queue + threads, the
run-once-config + live-hardware hybrid, the live diamond, property change
offline-vs-live, and cooperative stop). Read both before implementing.

```text
Phase: Live streaming, fake live source, threaded Queue, and the Sync element.

Goal: implement the live machinery so the Section 10-11 model runs, driven and
tested by a fake live source. Offline behavior must stay unchanged (120 tests
green).

Four finalized design decisions (docs/design/dataflow_sync_model.md S11):
  A. Default sync = per-slot Barrier (the existing vector-clock fold-match).
     Common-origin auto-synced; independent fires free. 99.99% needs no tuning.
  B. ONE counter per slot (generation). NO global (position,version) split --
     liveness (C) dissolves the conflation; any position/version need is internal
     to a Sync element only.
  C. Liveness-aware property change. Sources declare live/one-run (default
     one-run); propagate downstream by OR (live-driven = any input from a live
     source). SetProperty on a live-driven element applies FORWARD (no rerun,
     next buffer uses new config); on a one-run-driven element re-emits from cache
     (today's behavior). Mixed live+static => live-driven, static inputs Held.
  D. The Sync element is the SOLE front door for custom cross-source pairing.

Work items (design of record: dataflow_sync_model.md S10-S11, esp. S11.8):
  0. Unified run() = pump loop (S11.8). Generalize run() from one sweep to
     start_all -> pump-until-EOS -> stop_all. NO run_live(); NO OneShot/Streaming
     drive split. Live is NOT a run mode -- a one-run source ends after one buffer
     (offline unchanged, 120 tests green), a live source streams.
  0b. 3-state stream result (S11.8): Data / NoData (transient/timeout -> retry) /
     EndOfStream (terminal -> drains downstream, ends run()). nullopt/empty is
     reserved for NoData. A one-run source returns Data then EOS.
  0c. Cooperative stop (S11.8): std::stop_source on run(); stop()/SIGINT/window
     close call request_stop(); blocking source/Queue waits observe the stop_token
     (poll-with-timeout -> NoData ticks) and unwind promptly; graceful drain ->
     reverse stop_all -> return.
  1. Fake live-capture source: reads a Torch .pt file, emits ONE Buffer per axis-0
     entry ([50,5000] -> 50 buffers of [5000] or [1,5000]), then EOS; declares
     itself live; honors the stop token between rows. Deterministic fixtures, no
     hardware.
  2. Liveness model: source live/one-run flag (default one-run); link/start-time
     downstream OR-propagation; per-element live-driven flag; session SetProperty
     forward-vs-cache-rerun split. (Liveness is control-plane ONLY -- it does not
     pick the run path.)
  3. Segment-threaded execution + real Queue (S10): threads only at sources and
     Queues; Queue = rate-decouple + generation-boundary (QueueEpochPolicy) +
     thread boundary; aggregators fold-match across queue heads; no-drop /
     monotonicity on rejoining branches. (Threading is orthogonal to live: a
     finite file-backed live source streams single-threaded.)
  4. Sync element (S11.4): generic N->N; tuned policy (Zip/Barrier/Latest/Held/
     AllRequiredOnce); claims its own slot; stamps all N outputs of a fire alike
     (common-ancestor injection). Downstream uses default barrier.
  5. Join-mode enum (S11.1) promoted from contract-only to the Sync policy.

Out of scope: bundles / Mux / Demux (DROPPED -- superseded by the Sync element);
dedicated CPA score/rank plots; overlay plots; Kyber (Phase 30+).

Build/Test:
  CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
  cmake --build build -j && ctest --test-dir build --output-on-failure

Completion checklist:
  - All existing (offline) tests still green; offline behavior unchanged.
  - Fake live source emits per-axis-0-row buffers from a fixture.
  - Liveness propagates; a live-driven property change applies forward (no
    rerun), a one-run-driven one re-emits from cache.
  - Threaded Queue handoff works; segment threading = components cut at Queues.
  - A Sync element pairs two fake live streams; a downstream default-barrier join
    consumes the aligned outputs with no per-pad tuning.
```

## Current Context Task

Phase 22 has added the ImGui/ImPlot plotting foundation:

- `leakflow_plot` owns plot snapshots, grouping state, and the OpenGL3 GUI loop.
- `TracePlot` is sink-only and registers plot-owned snapshots.
- `TorchConvert` explicitly prepares Torch tensors for plotting.
- `leakflow run` enters the plot loop only after synchronous pipeline execution
  and only when plot sessions exist.
- The tutorial app is optional via `LEAKFLOW_BUILD_EXAMPLES=ON`.

Phase 23 has added `leakflow_crypto`:

- scalar and Torch `uint8` Hamming weight helpers,
- scalar and Torch `uint8` Hamming distance helpers,
- AES S-box helpers,
- AES first-round S-box leakage helpers returning `m`, `y`, `HW(m)`, and
  `HW(y)`,
- AES first-round leakage-channel helpers for known-key `[U,N,L]` leakage and
  guess-domain `[U,G,N,L]` hypotheses,
- helper-library only; crypto plugin elements arrived in Phase 24.

Phase 24 has added `leakflow_plugins_crypto`:

- `AesLeakage` for AES Hamming-weight leakage targets,
- `AesLeakageHypothesis` for AES Hamming-weight key-guess leakage hypotheses,
- `PearsonPoiFinder` with real named `features` and `targets` sink pads,
- `CorrelationPoiPayload` for correlation PoI results,
- target-label, AES byte-index, and channel metadata such as
  `poi.target.0.label`, `poi.target.0.byte_index`, and
  `poi.target.0.channel`,
- `CorrelationPoiToPlotAnnotations` to convert correlation PoIs into generic
  `PlotAnnotationPayload` markers.

The AES CPA/report slice has added:

- `CpaAttack` for generic Pearson CPA scoring/ranking,
- `AttackStats` for known-key true-rank/success diagnostics,
- `AttackStatsToPlotAnnotations` for marking stats-backed attack best samples on
  `TracePlot`.

Early AES plotting support has added:

- `PlotAnnotationPayload`,
- optional `TracePlot.annotations` sink,
- TracePlot snapshot annotation markers for selected sample indexes.

Post-Phase 24 metadata refinement has added:

- pad-targeted metadata descriptors,
- `leakflow-ls` metadata grouping by all pads, exact pads, and `%u` pad
  templates,
- `leakflow run` metadata annotations for all output pads, exact output pads,
  and matching `%u` output pad templates.

Post-Phase 24 graph-inspection support has added:

- `PipelineObserver` and `PipelineEvent` copied snapshots in core,
- topology snapshots for element type/name/kclass, properties, pads, and links,
- routed-buffer observations carrying caps, metadata, payload presence/type,
  sequence numbers, and no raw payload data,
- `PipelineGraphRuntime` and an ImGui graph renderer in `leakflow_plot`,
- `leakflow run --graph`, which runs the synchronous pipeline in a worker
  thread while the main UI loop draws the graph and any `TracePlot` sessions.

Post-Phase 24 graph/control support has added:

- reusable `run_pipeline_graph_until_closed(...)` for linked applications,
- `PipelineControlRuntime` as an explicit control plane separate from copied
  graph observation,
- property controls generated from `PropertySpec` / `PropertyValue`,
- graph gear buttons and standalone controls panels,
- `PropertyEffect` declarations for `ui-only`, `sink-display`,
  `metadata-output`, `payload-output`, `caps-output`, and `lifecycle` changes,
- copied `PropertyChanged` observer events,
- `AesLeakage.channels` declared as `payload-output` with downstream
  invalidation on `leakage`.

Important graph/control design decisions:

- Graph observation is copied and SCA-safe; it must not expose mutable elements,
  mutable buffers, payload pointers, raw traces, keys, or plaintext arrays.
- Control is an explicit live mutation path. Current controls may mutate
  properties, but the next phase should replace direct mutation with a
  `PipelineController` / `PipelineSession` command API.
- Element properties are the source of truth for property-backed UI controls.
  Graph gear controls, global control panels, and future element-local controls
  should stay synchronized through the same property/control path.
- Display-only changes do not create new buffers. Caps, metadata, payload, and
  lifecycle changes are dataflow changes and need controller/session handling.
- Metadata-output changes should create a new `Buffer` envelope when rerun, not
  bypass downstream elements through a side channel.
- Live pipelines should use config epochs. A live `Queue` must eventually make
  drain/flush/keep-mixed epoch policy explicit.

Phase 25 has added the control/session layer (see
`docs/design/pipeline_controller.md`):

- `PipelineSession` in `leakflow_core` owns a `Pipeline` (move-in) and is
  the application's single observe-and-control handle.
- `Pipeline` gained `start_all`/`run_sweep`/`rerun_from`/`stop_all`
  primitives and a per-input-pad / per-output-pad buffer cache; `run()` is sugar
  over them; the dead recursive `run_from` was removed.
- (Phase 25 used `Buffer::epoch()` as the configuration-generation stamp;
  **Phase 27 removed it** — buffer generation/sync is now the vector clock
  (`Buffer::provenance()`), and the session keeps a monotonic `generation`
  counter.)
- `Element::can_replay()` (default true, `Queue` false, mirrored into the
  descriptor for `leakflow-ls`) drives partial-rerun-vs-full-restart escalation;
  `Queue::start()` now clears its buffers.
- UI submits `SetProperty` to a thread-safe, last-wins-coalescing command queue;
  the worker applies commands at safe points (between units of work) and reports
  copied `CommandAccepted/Rejected/Applied` events on the observer stream.
- caps-output changes are validated transactionally; rerun failures do not tear
  down the session.
- The graph runner uses a persistent worker loop with a pluggable drive policy
  (`OneShotDrive` now). `StreamingDrive`, threaded `QueueEpochPolicy` enforcement,
  cooperative cancel, `Paused`/pause/resume, and preroll are reserved seams.
- `PipelineControlRuntime` is now a UI client of the session;
  `run_pipeline_graph_until_closed` takes a `PipelineSession&`; graph control
  buttons map to re-run-from-sources, stop-start cycle, and live start/stop.

## Phase Brief Template

When starting a new phase, update or temporarily replace this section with:

```text
Requested phase:

Goal:

Allowed modules/files:

Out of scope:

Expected changed files:

Build commands:

Test commands:

Completion checklist:
```

## Default Out Of Scope

Unless the requested phase explicitly says otherwise:

- no AES,
- no crypto plugins,
- no Kyber,
- no new GUI,
- no YAML,
- no dynamic plugin loading,
- no CUDA behavior changes,
- no new external dependencies,
- no unrelated refactors.

## Default Reporting Checklist

After editing files, report:

- changed files,
- build commands,
- test commands,
- whether tests were actually run,
- whether the requested phase/task is complete,
- next recommended phase.
