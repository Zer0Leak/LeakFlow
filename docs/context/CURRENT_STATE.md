# Current Project State

This is the compact state summary for LeakFlow. Prefer this over reading every
root Markdown file at session start.

## Identity

LeakFlow is a C++23 side-channel analysis framework with a pipeline/plugin
architecture inspired by GStreamer. It is for reproducible SCA experiments, not
a generic machine-learning framework.

Primary validation target: AES.

Primary later research target: Kyber / ML-KEM.

## Phase State

The repository is post-Phase 27 (DAG executor, per-pad outputs, and vector-clock
buffer provenance; `Buffer::epoch()` removed; `LinearPipeline` renamed to
`Pipeline`) **with the live-streaming phase implemented** (threaded segments, the
real `Queue`/`BufferQueue`, the `Sync` element, `FakeLiveSrc`, the liveness model,
cooperative stop, and the `Stopped/Running/Paused/Idle` player state machine — see
`docs/design/dataflow_sync_model.md` §12–14), plus early AES plotting annotation
support:

- Modular `include/leakflow/...` public include tree is in place.
- Core, render, base, extras, apps, core plugins, base plugins, and extras
  plugins are split into separate source directories and CMake targets.
- Structured buffer/payload summaries and terminal rendering are implemented.
- `NumpySrc` exists and can load `.npy` files into a `NumpyPayload`.
- A format-neutral `TensorDatasetReader` contract and the
  `Hdf5TensorDatasetReader` backend exist in `leakflow_extras`. HDF5 arrays are
  discovered with their attributes and loaded into preallocated Torch tensors
  through row hyperslabs with logical-byte progress.
- A direct application-callable NumPy-to-Torch conversion API exists in
  `leakflow_extras`.
- `NumpyToTorch` exists and exposes the NumPy-to-Torch conversion through the
  extras plugin.
- `TorchFileSrc` and `TorchFileSink` exist and load/save single Torch tensor `.pt`
  files through the base plugin.
- `Hdf5FileSrc` exists in the extras plugin. It exposes optional named
  `traces`, `plaintexts`, `keys`, `ciphertexts`, and `countermeasures` outputs,
  internally reads in hyperslab batches with progress, and emits complete Torch
  tensors/bundles.
- `FakeLiveHdf5Src` replays aligned HDF5 row batches at an optional trace rate
  and reports determinate trace-count progress.
- `leakflow_log` exists and wraps spdlog behind a LeakFlow-owned logging API.
- `leakflow_plot` exists and provides the ImGui/ImPlot plot runtime.
- `leakflow_crypto` exists and provides Hamming weight/distance helpers plus
  AES first-round S-box leakage helpers, including `y(0)`..`y(7)` S-box output
  bit channels.
- `leakflow_ml` exists and provides generic Torch numeric APIs for Gaussian
  mixtures, constrained Sinkhorn transport, the legacy scalar-class clustering
  metrics, and the complete Phase A vector-truth exact, semantic,
  fragmentation, rectangular-alignment, and optional combined-quality
  evaluator.
- `leakflow_plugins_ml` exists and provides `FeatureSelect`, `GaussianMixture`,
  `ClusteringStats`, and the structured `ClusteringEvaluate` /
  `ClusteringEvaluationPayload` boundary, including typed-unit
  propagation/alignment.
- `leakflow_plugins_ml_plot` exists and provides the bounded
  `ClusteringMetricsTablePlot` bridge. It fills domain-free `TableView` tab
  groups with Overview, Exact, Semantic, Fragmentation, Combined, Alignment,
  Heatmap, and Parameters tabs.
- `leakflow_plugins_crypto` exists and provides AES S-box leakage, AES
  guess-domain leakage hypotheses, generic Pearson CPA ranking, generic DPA
  difference-of-means ranking with an optional best-difference trace output,
  generic known-key attack stats, attack stats plot-annotation conversion,
  Pearson PoI finding, and
  correlation-PoI-to-plot-annotation conversion elements.
- `leakflow_plugins_plot` exists and provides sink-only `TracePlot` and
  `HeatmapPlot`.
- `TracePlot` accepts optional generic plot annotations and renders selected
  sample markers.
- `PipelineObserver` and copied graph snapshots exist in `leakflow_core` for
  topology, property, pad, link, and routed-buffer observations.
- `leakflow_plot` provides an ImGui pipeline graph runtime/renderer that
  consumes core observer events without storing payload pointers or raw trace
  values.
- `PipelineControlRuntime` exists in `leakflow_plot` as an explicit live-control
  path separate from graph observation. It weakly binds to live elements,
  renders property controls from `PropertySpec` / `PropertyValue`, supports graph
  gear buttons and standalone panels, records `PipelineControlChange`, and emits
  copied `PropertyChanged` events.
- `PropertyEffect` exists on `PropertySpec` to describe property change behavior:
  `ui-only`, `sink-display`, `metadata-output`, `payload-output`, `caps-output`,
  and `lifecycle`, with invalidation scope and optional affected output pads.
- `PipelineSession` (Phase 25) in `leakflow_core` owns a `Pipeline` and is
  the application's single observe-and-control handle. It owns lifecycle
  (`start_all` once → `run_sweep`/`rerun_from` many → `stop_all`), a thread-safe
  `SetProperty` command queue with last-wins coalescing, safe-point command
  application, transactional caps validation, downstream-only rerun with
  escalation, rerun error handling (no session teardown), a monotonic
  `generation` counter (Phase 27; renamed from epoch), a
  `Stopped/Started/Running` state machine (`Paused` reserved), and session
  controls (restart, re-run-from-sources, caching toggle). It emits copied
  `CommandAccepted/Rejected/Applied` events on the existing observer stream. See
  `docs/design/pipeline_controller.md`.
- `Pipeline` (renamed from `LinearPipeline` in Phase 27) exposes
  `start_all`/`run_sweep`/`rerun_from`/`stop_all` primitives plus a
  per-input-pad / per-output-pad buffer cache; `run()` is sugar over them for
  one-shot callers and engine tests. Elements produce per-pad `ElementOutputs`
  via `process_pads(...)`; `Tee` owns fan-out explicitly (no implicit engine
  broadcast); the executor routes per-pad.
- Buffer synchronization uses a **vector clock** (Phase 27), not an epoch.
  `Buffer::provenance()` is a dense per-element production-count clock; `Pipeline`
  allocates a slot per producer at `add()` (`ElementDescriptor::provenance_slots`;
  `Tee`/sinks claim 0); the executor increments the producer's slot on emission
  and runs `merge_provenance()` (conflict-detecting fold-match) at joins. Partial
  rerun is now verified by the fold-match; per-buffer "generation" for UI/observer
  is `provenance_generation()` (max of the clock). `Buffer::epoch()` was removed.
  See `docs/design/dataflow_sync_model.md`.
- `Element::can_replay()` (default true, mirrored into `ElementDescriptor`)
  signals replay safety; `Queue` returns false and clears its buffers in
  `start()`.
- The live-streaming machinery is **implemented**: `Pipeline::run()` is a unified
  pump loop (no `run_live()`; live auto-detected via a live source), threaded
  segments are cut at every `Queue` (`pipeline_segments`, `run_threaded`), `Queue`
  is a thread-safe `BufferQueue` with Block/DropOldest/DropNewest policy, the
  aggregator fold-matches queue heads (Barrier/Held/Latest), `Sync` (`N→N`) injects
  a common ancestor for independent streams, cooperative stop flows through a
  `std::stop_token` (CLI SIGINT + `--graph` window close), and the session owns a
  `Stopped/Running/Paused/Idle` player state machine with a safe-point control
  plane. `QueueEpochPolicy` remains a documented enum; the generation-boundary
  drain/flush knob is intentionally optional because the vector clock matches by
  ancestor identity (config-independent). Preroll remains the only unbuilt seam.
- `AesLeakage.channels` is declared as `payload-output` with downstream
  invalidation on the `leakage` output pad.
- Metadata descriptors can target all pads, exact pads, or `%u` pad templates;
  `leakflow-ls` groups metadata by those pad targets.
- Every element exposes a common readable/writable string `name` property.
  Default generated CLI names use the lower-case alphanumeric element type plus
  a zero-based index, such as `fakesrc0`, `tee0`, and `tee1`.
- `Pipeline` owns added-element name uniqueness, locks element names after
  add, and can look up elements by instance name or descriptor type.
- `ElementFactoryRegistry` pairs linked plugin descriptors with explicit element
  factories. The `leakflow_cli` helper can build a pipeline from the same
  expression language used by `leakflow run`.
- `leakflow run` metadata annotations can apply to all output pads, one output
  pad, or matching output pads through a `%u` pad-template selector.
- `TorchConvert` exists in `leakflow_plugins_base` for explicit
  `TorchTensorPayload -> TorchTensorPayload` dtype/device conversion.
- The `leakflow` CLI supports `--log-level`, `--log-color`, `--log-filter`,
  `--summaries`, `--no-summaries`, `--summary-level` (payload detail 0-3 in
  `--graph` tooltips and the info buffer-flow log; default 2), per-run
  `--telemetry` / `--no-telemetry` (default off headless, on with `--graph`), and
  the profiling flags `--print-profile` (per-element timing table at exit) and
  `--profile-file PATH` (Chrome trace JSON for chrome://tracing / Perfetto). See
  `docs/design/profiling.md`.
- `leakflow run` enters the plot GUI loop after synchronous pipeline execution
  when one or more plot sessions were registered.
- `leakflow run --graph` runs the synchronous pipeline in a `std::jthread`
  worker and draws the live pipeline graph plus any `TracePlot` sessions in the
  main ImGui loop.
- LeakFlow-owned logging environment variables are supported:
  `LEAKFLOW_LOG_LEVEL`, `LEAKFLOW_LOG_COLOR`, `LEAKFLOW_LOG_FILTER`,
  `LEAKFLOW_SUMMARIES`, and `LEAKFLOW_SUMMARY_LEVEL`.
- `Summary(always_print=true)` overrides disabled global summaries.
- An optional tutorial app target can be built with
  `-DLEAKFLOW_BUILD_EXAMPLES=ON`.
- Metadata forwarding is policy-driven: a four-group taxonomy
  (`capture`/`origin`/`payload`/`routing`) plus per-element forwarding profiles
  derived from the `<Profile>/<Family>[/<Role>[/<Variant>...]]` klass scheme. See
  `docs/design/metadata_klass_taxonomy.md`.

Phase 25 (control/session layer) is implemented. The full AES PoI plotting
pipeline (`Hdf5FileSrc` → `Tee`/`AesLeakage` → `PearsonCorrelator` → `PoiSelect` →
`CorrelationPoiToPlotAnnotations` → `TracePlot`) is built and runs end to end:
manually validated in `leakflow run --graph`, with headless CLI smoke tests in
`tests/apps` (`leakflow_cli_run_aes_leakage`, `leakflow_cli_run_pearson_poi_finder`,
`leakflow_cli_run_correlation_poi_to_plot_annotations`) asserting metadata/caps
flow.

Phase 26 (AES PoI pipeline correctness validation) is implemented. A headless
numeric test (`tests/plugins/crypto/aes_poi_correctness_test.cpp`, CTest
`leakflow_plugins_crypto_aes_poi_correctness`) loads the checked-in fixtures
(`key_01`, `key_02`) via the same `torch::pickle_load` path the pipeline uses and
asserts: `AesLeakage` HW(m)/HW(y) for a known byte match values computed directly
from the fixture plaintext/key (raw S-box table + `std::popcount`, plus the
`leakflow_crypto` scalar helper); `PoiSelect` selects the
strongest-correlation sample with a correlation matching an independent float64
recompute to 1e-6 and clearing a sane threshold; and
`CorrelationPoiToPlotAnnotations` sample indexes/values/precision formatting match
the PoI output. Tests have no dependency on the ignored local `traces/` tree.

Phase 27 (DAG executor, per-pad outputs, vector-clock buffer provenance;
`Buffer::epoch()` removed; `LinearPipeline` → `Pipeline`) is implemented; all
120 tests green. Sync between branches is now established by the buffer vector
clock (fold-match at joins), which made bundles/`Mux`/`Demux` (the old 27.4/27.5)
unnecessary offline — they moved to the live phase. See
`docs/design/dataflow_sync_model.md`.

The **live-streaming phase is implemented** (offline behavior unchanged), built
on the **finalized synchronization & liveness design**
(`docs/design/dataflow_sync_model.md` Sections 10–14). Four locked decisions, all
now wired:

- **A. Default sync = per-slot Barrier** (the existing vector-clock fold-match) —
  common-origin auto-synced, independent fires free; 99.99% of pipelines need no
  tuning. The threaded aggregator runs this fold-match across queue heads.
- **B. One counter per slot** (generation); **no global (position, version)
  split** — liveness (C) dissolves the need.
- **C. Liveness-aware property changes** — sources declare live/one-run (default
  one-run), propagated downstream by OR; `SetProperty` on a live-driven element
  applies *forward* (next buffer), on a one-run-driven element re-emits from cache.
- **D. The `Sync` element** (generic `N→N`) is the **sole front door** for custom
  cross-source pairing: it injects a common ancestor so downstream uses the
  default barrier. Bundles/`Mux`/`Demux` are **dropped** (superseded).

Delivered: `FakeLiveSrc` (reads a Torch `.pt`, emits one `Buffer` per axis-0 row,
`trace_rate` pacing, then EOS), the liveness model, threaded segments + the
real `Queue`/`BufferQueue`, the `Sync` element and its policies, cooperative stop,
and the `--graph` player controls (Start/Stop/Pause/Resume, Auto-apply). See
`docs/design/dataflow_sync_model.md` §12 (implementation map + tests), §13 (player
state machine), and §14 (CLI cookbook). Full clustering evaluation Phase A is
implemented: A1 exact numeric core, A2 semantic/fragmentation metrics, A3
rectangular alignments, and A4 pipeline/table inspection are complete. Payload
persistence, generic metric charts, and a standalone/selectable clustering
matrix plot are unblocked but remain deferred. A user-requested bounded post-A4
extension adds explicit N/S comparison columns and a same-window stored-
contingency Heatmap tab. Zarr parity remains a separate deferred candidate.

Generic `Convert`, the conversion registry, and conversion-registry dynamic pads
remain deferred as low-priority future infrastructure.

## Existing Targets

- `leakflow_core`: framework kernel.
- `leakflow_log`: spdlog-backed logging layer.
- `leakflow_render`: terminal styling and summary rendering.
- `leakflow_base`: LibTorch-backed tensor payload layer.
- `leakflow_crypto`: LibTorch-backed crypto/SCA leakage helper layer.
- `leakflow_ml`: generic Torch clustering, mixture, transport, and evaluation
  numeric layer; no core dependency.
- `leakflow_extras`: format-neutral tensor-dataset reader with an HDF5 backend,
  plus NumPy payload/loading and NumPy-to-Torch conversion.
- `leakflow_plot`: ImGui/ImPlot plotting runtime and sessions.
- `leakflow_plugins_core`: linked shared plugin with generic elements.
- `leakflow_plugins_base`: linked shared plugin with Torch-backed elements.
- `leakflow_plugins_extras`: linked shared plugin with `Hdf5FileSrc`,
  `FakeLiveHdf5Src`, `NumpySrc`, `NumpyToTorch`, and `BufferFileSink`/`BufferFileSrc`.
- `leakflow_plugins_crypto`: linked shared plugin with `AesLeakage`,
  `AesLeakageHypothesis`, `CpaAttack`, `DpaAttack`, `AttackStats`,
  `AttackStatsToPlotAnnotations`, `PearsonCorrelator`, `PoiSelect`, and
  `CorrelationPoiToPlotAnnotations`.
- `leakflow_plugins_ml`: linked shared plugin with `FeatureSelect`,
  `GaussianMixture`, and `ClusteringStats`.
- `leakflow_plugins_plot`: linked shared plugin with `TracePlot` and
  `HeatmapPlot`.
- `leakflow_cli`: static CLI helper library.
- `leakflow`: main CLI executable.
- `leakflow-ls`: linked descriptor inspection executable.
- `leakflow_cuda_smoke`: optional CUDA smoke executable when
  `LEAKFLOW_WITH_CUDA=ON`.
- `leakflow_trace_plot_tutorial`: optional tutorial executable when
  `LEAKFLOW_BUILD_EXAMPLES=ON`.

## Implemented Concepts

Core:

- `Element` identity fields for logging:
  - element type,
  - element instance name,
  - element kclass.
- `Element` synchronizes its instance name with the common string `name`
  property.
- `Caps`
- `Buffer`
- `Payload` with a required non-empty logical `layout()`; attaching one through
  `Buffer::set_payload(...)` replaces the reserved `payload.layout` metadata,
  while clearing it removes that metadata.
- `Element`
- `Pad`
- `Pipeline`
- `ElementFactoryRegistry`
- `PayloadCodecRegistry` (payload save/load callbacks by `type_name`, writing
  through a storage-neutral `BufferArchiveWriter`/`Reader`; drives
  `BufferFileSink`/`BufferFileSrc`. Core holds only the callbacks and
  forward-declares the archive types, so it stays Torch/HDF5-free)
- `PropertyValue` / `PropertySpec`
- `PropertyEffect`
- `ElementDescriptor` / `PluginDescriptor`
- `MetadataPadTarget` / `ElementMetadataDescriptor` pad targets
- `SummaryDocument`
- `PipelineObserver` / `PipelineEvent` copied graph-observation snapshots
- copied `PropertyChanged` observations for control-plane property edits
- per-pad `ElementOutputs` / `Element::process_pads(...)`; explicit `Tee` fan-out
- `Buffer::provenance()` vector clock; `provenance.hpp`
  (`merge_provenance` / `provenance_compatible` / `provenance_generation`);
  `ElementDescriptor::provenance_slots`
- `PipelineSession` control/session layer with `SetProperty` command queue,
  safe-point application, cached buffers, downstream-only rerun, monotonic
  `generation` counter, state machine, and copied command events
- `Element::can_replay()` replay-safety signal
- `Element::is_live()` / `at_end_of_stream()` / `set_stop_token()`; 3-state stream
  result (Data / NoData / EndOfStream)
- `BufferQueue` (thread-safe bounded FIFO; Block/DropOldest/DropNewest) and
  `pipeline_segments` decomposition (`run_threaded`, one `std::jthread` per segment)
- threaded aggregator (Barrier fold-match across queue heads, realign-on-drop,
  Held, Latest); live `Pipeline::run()` pump loop; cooperative stop
- `PipelineSession` `Stopped/Running/Paused/Idle` state machine, pause/resume, and
  safe-point forward-apply control plane
- cooperative checkpoints for pausing long synchronous elements; Stop suppresses
  late outputs, and progress distinguishes Active/Completed/Cancelled outcomes
- `QueueEpochPolicy` enum (documented; drain/flush generation-boundary knob is an
  optional policy, not a correctness requirement)
- `MetadataGroup` / `ForwardingProfile` / `metadata_group` / `profile_for_klass`
  / `forward_metadata` metadata forwarding policy
- Limited linked caps propagation through generic forwarding elements.
- Timing telemetry / profiling (`TelemetryKind::Duration`,
  `RuntimeTelemetryDurationStat`, `RuntimeTelemetryScopedTimer`,
  `TelemetryTraceSink`): the executor times each element's `process` step; elements
  add opt-in op scopes via `Element::profile_scope`; gated separately from size
  telemetry (profiling defaults off). Surfaced by `--print-profile` (table),
  `--profile-file` (Chrome trace, one track per element), and a live `--graph`
  timing overlay (per-element duration telemetry published to the graph panel).
  `Queue` profiles wait time (`backpressure` / `starvation`).
  Design: `docs/design/profiling.md`.

Base:

- `TorchTensorPayload`
- `TorchTensorBundlePayload`
- `PlotAnnotationPayload`
- `pearson_correlation(...)`
- `BufferArchiveWriter` / `BufferArchiveReader` (storage-neutral, Torch-aware
  payload serialization interface used by the payload codecs; HDF5-agnostic)

Extras:

- `TensorDatasetReader` / `TensorDatasetDescriptor`
- `Hdf5TensorDatasetReader`
- `Hdf5BufferArchiveWriter` / `Hdf5BufferArchiveReader` (HDF5 backend for the
  `leakflow.buffer` whole-Buffer format: envelope attributes + native payload
  datasets; the concrete `BufferArchiveWriter`/`Reader` implementation)
- `NumpyPayload`
- `load_npy(path)`
- `convert_numpy_to_torch(...)`

Crypto:

- Scalar and Torch `uint8` Hamming weight helpers.
- Scalar and Torch `uint8` Hamming distance helpers.
- AES S-box helper.
- AES first-round S-box leakage helpers that return `m`, `y`, `HW(m)`, and
  `HW(y)` for scalar bytes or Torch tensors.

ML:

- `GaussianMixture`: batched full/diagonal-covariance EM with multiple
  initializations and optional size-constrained Sinkhorn assignment. Its labels
  output reports the fitted feature width as `payload.cluster.n_features`.
- `sinkhorn(...)`: generic log-domain entropic optimal transport.
- Current `clustering_metrics` APIs: confusion matrix, purity, ARI, arithmetic
  NMI, square Hungarian matching/reordering, and matched accuracy/per-class
  precision/recall/F1.
- `evaluate_clustering(...)`: core-free Phase A1 evaluation for arbitrary
  int64-representable labels `[N]`/`[U,N]` and exact vector truth
  `[N,D]`/`[U,N,D]`. It returns deterministic sparse contingency detail, checked
  unordered-pair counts, ARI, arithmetic AMI, homogeneity, completeness,
  V-measure, purity, pair precision/recall/F1, and arithmetic NMI with explicit
  value/support/undefined semantics.
- Phase A2 extends that result with explicit `off|power` semantic evaluation,
  strict ranges/weights and `power=1|2`, semantic impurity micro/macro and
  per-dimension/per-cluster records, merge-error rate and conditional severity,
  and fragmentation micro/macro and per-group records. Global detail stays
  bounded; Full detail retains singleton clusters/groups as explicit unavailable
  records. The `p=1` and `p=2` kernels avoid observation-pair enumeration.
- Phase A3 adds optional `none|exact|semantic|both` alignment. Exact-overlap and
  semantic-cost mappings remain separate, support rectangular `G<K` and `G>K`,
  use strict predicted-major dense tie-breaking, expose unmatched support on
  both marginals, and materialize canonical mapping identities even in Global
  mode. Full detail adds exact per-truth scores and semantic contingency-mass
  error records; semantic alignment uses the configured power cost and a fixed
  maximum dummy penalty of `1.0`.
- Phase A4 added the default-off combined-quality record,
  `ClusteringEvaluationPayload`, and `ClusteringEvaluate`. The element aligns
  typed units carried by its input/output `Buffer`s, stores effective
  `evaluation.*` options plus bounded labels-side `payload.cluster.*` producer
  parameters in the payload, and emits a bounded summary. The payload itself
  does not own unit identity.

ML plugin elements:

- `FeatureSelect`
- `GaussianMixture`
- `ClusteringStats`: accepts scalar truth IDs, aligns units, and emits a dense
  reordered confusion tensor with ARI/NMI/purity/matched-score metadata. This is
  the current/legacy matrix evaluator; it does not yet accept semantic vectors
  or emit a structured evaluation result.
- `ClusteringEvaluate`: accepts `labels` and vector `truth`, emits structured
  clustering evaluation on `evaluation`, and keeps `ClusteringStats` unchanged.

ML plot bridge:

- `ClusteringMetricsTablePlot`: consumes that result on `sink`. The original A4
  bridge supplied the metric/parameter tables; a user-requested bounded post-A4
  extension added the explicit N/S columns and same-window Heatmap described
  here. Overview has one
  row per run and unit with explicit `Observations (N)` and `Features (S)` shape
  columns, counts, headline metrics, and core producer plus experiment
  parameters. `Features (S)` reads captured `labels.cluster.n_features` and is
  `N/A` for producers that do not report it. Exact, Semantic, Fragmentation,
  Combined, and Alignment contain every stored `MetricValue` exactly once, while Parameters
  presents effective/captured parameters once per run, retaining
  `labels.cluster.n_features`. Heatmap is the eighth tab: it consumes only stored
  Full-detail sparse contingency, applies the stored exact-overlap column
  permutation when present (otherwise raw order), row-normalizes copied counts,
  and labels rows from truth vectors/dimension names and columns from actual
  predicted IDs. It shares typed-unit selection, supports ragged per-unit shapes,
  and caps dense display at a combined 1,000,000 cells per run across unit
  pages; Global detail and oversized frames show an unavailable reason. Metric
  labels use `↑` or
  `↓` for their optimization direction. Accumulate/replace, synchronized typed-
  unit selection, clear, tab-local stable column sorting, and independent
  per-run table/heatmap history operate only on copied display data and never
  recompute evaluation or assignment. `update_mode` is
  `UiControl`/`ElementUi` with `auto|accumulate|replace` and read-only
  `active_update_mode`; auto follows liveness. Group/title and view-local
  unit/tab/clear/sort behavior are `UiControl`.

Core plugin elements:

- `FileSrc`
- `FileSink`
- `FakeSrc`
- `FakeSink`
- `Summary`
- `Tee`
- `Queue`
- `Sync` (generic `N→N` cross-source pairing; barrier/zip/all-required-once/held/
  latest policies)

Base plugin elements:

- `TorchFileSrc`
- `TorchConvert`
- `TorchFileSink`
- `FakeLiveSrc` (live `.pt` source: one buffer per axis-0 row, `trace_rate`
  pacing, then EOS)

Extras plugin elements:

- `Hdf5FileSrc`
- `FakeLiveHdf5Src`
- `NumpySrc`
- `NumpyToTorch`
- `BufferFileSink` / `BufferFileSrc` (persist/reload a whole Buffer to a single
  HDF5 file — the `leakflow.buffer` schema: envelope as attributes, payload body
  written as native datasets by the `PayloadCodec` registered for the payload's
  `type_name()` through a `BufferArchiveWriter`/`Reader`)

Plot plugin elements:

- `TracePlot` with an optional `annotations` sink for
  `leakflow/plot-annotations`.

Crypto plugin elements:

- `AesLeakage`
- `AesLeakageHypothesis`
- `CpaAttack`
- `DpaAttack`
- `AttackStats`
- `AttackStatsToPlotAnnotations`
- `PearsonCorrelator`
- `PoiSelect`
- `CorrelationPoiToPlotAnnotations`

`CorrelationPayload` uses the generic dense layout
`unit/channel/feature`. `CorrelationPoiPayload` is organized by unit, with each
unit carrying `channel/poi/[sample_index,score]` (or score-only) results. Generic
correlation/PoI APIs and archive fields use `unit_indexes`; AES-specific byte
selection remains confined to the AES leakage elements.

Logging:

- `LogLevel`
- `LogColorMode`
- `LogRecord`
- `LogFilter`
- `LogConfig`
- CLI/env configuration with CLI precedence over environment variables.

## Not Implemented Yet

- Deferred clustering-evaluation work: versioned payload persistence, generic
  `MetricView`/`ClusteringMetricsPlot`, and the standalone
  `ClusteringMatrixPlot` with selectable raw/exact/semantic matrices and
  normalization. The bounded row-normalized Heatmap tab is already implemented
  as the post-A4 extension and does not provide those selectable modes. Design
  of record: `docs/design/clustering_evaluation_metrics.md`.
- Zarr tensor-dataset reader/source/converter and the HDF5/Zarr benchmark.
- Pipeline `Convert` element.
- Conversion registry.
- Dynamic pads.
- Dynamic plugin loading.
- Torch bundle `.pt` / `.pth` I/O (HDF5 dataset bundles are supported through
  named source pads).
- TorchScript module I/O elements such as future `TorchScriptSrc` and
  `TorchScriptSink`.
- Kyber helpers or Kyber plugins.
- Generic SCA statistics/PoI/labeling plugins.
- Headless plot rendering tests.
- Dynamic plot sink pads.
- A standalone `leakflow-plot` executable.
- YAML/config runner.
- Kyber / ML-KEM hypothesis elements for the generic CPA attack path.
- Dedicated CPA score/rank/correlation/heatmap plot elements (the first CPA
  plot bridge is `AttackStatsToPlotAnnotations` for `TracePlot.annotations`).
- Overlay / correlation plot elements beyond the existing annotation overlays.
- `QueueEpochPolicy` drain/flush enforcement (optional policy; not required for
  correctness) and the `preroll` player refinement.

## Dependencies

- C++23.
- CMake with Ninja preferred.
- Clang preferred.
- LibTorch is required from Phase 15 onward.
- `fmt` is fetched by CMake for render formatting/color.
- `spdlog` is fetched by CMake for logging.
- `cnpy++` is fetched by CMake for extras NumPy support.
- Dear ImGui and ImPlot are fetched by CMake for plotting.
- GLFW, OpenGL, and zlib development files are required for plotting.
- Boost filesystem/iostreams are required by `cnpy++`.
- HDF5 is required by the extras tensor-dataset backend.
- CUDA is optional and gated by `LEAKFLOW_WITH_CUDA`.

## Test Fixtures

The current checked-in AES dataset fixture is:

- `tests/fixtures/aes/sync/key_01.h5`, with `/traces` (`float32 [50,5000]`),
  `/plaintexts` (`uint8 [50,16]`), `/keys` (`uint8 [16]`), and
  `/ciphertexts` (`uint8 [50,16]`).

Focused Torch file-I/O and Phase 26 correctness tests retain the original tiny
Torch fixtures under `tests/fixtures/aes/sync/`:

- `key_01/traces_first_50.pt`: `float32`, shape `(50, 5000)`.
- `key_01/plain_texts_first_50.pt`: `uint8`, shape `(50, 16)`.
- `key_01/key_first_50.pt`: `uint8`, shape `(16,)`.
- `key_02/traces_first_50.pt`: `float32`, shape `(50, 5000)`.
- `key_02/plain_texts_first_50.pt`: `uint8`, shape `(50, 16)`.
- `key_02/key_first_50.pt`: `uint8`, shape `(16,)`.

Tests must use these fixtures instead of depending on the ignored local
`traces/` tree.

## Default Commands

```bash
CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/.local/lib/libtorch"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful CLI smoke commands:

```bash
./build/leakflow --help
./build/leakflow --log-level debug --log-color never --log-filter element=FakeSrc run 'FakeSrc ! FakeSink'
./build/leakflow --no-summaries run 'FakeSrc ! Summary(always_print=true)'
./build/leakflow run 'FakeSrc ! Summary'
./build/leakflow run --graph 'FakeSrc ! Tee@t; @t.src_0 ! Summary; @t.src_1 ! FakeSink'
./build/leakflow run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! Summary ! FakeSink'
./build/leakflow run 'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/traces_first_50.pt) ! TorchFileSink(path=/tmp/traces_first_50_roundtrip.pt)'
./build/leakflow run 'Hdf5FileSrc@data(path=tests/fixtures/aes/sync/key_01.h5); @data.traces ! TracePlot(title="AES traces",group=aes,label=trace)'
./build/leakflow-ls
./build/leakflow-ls TracePlot
```
