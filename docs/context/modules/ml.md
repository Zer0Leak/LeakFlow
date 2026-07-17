# ML Module Context

Use this for the generic (domain-free) machine-learning / statistics numeric layer:
clustering, mixture models, optimal transport, and clustering evaluation. No SCA or
AES knowledge lives here.

User-facing metric interpretation, alignment guidance, and Heatmap reading are
documented in
[`docs/guides/CLUSTERING_EVALUATION.md`](../../guides/CLUSTERING_EVALUATION.md).

## Files

- Numeric public headers: `include/leakflow/ml`
- Numeric sources/tests: `src/ml`, `tests/ml`
- Pipeline public headers: `include/leakflow/plugins/ml`
- Pipeline sources/tests: `src/plugins/ml`, `tests/plugins/ml`
- Table-bridge headers/sources/tests: `include/leakflow/plugins/ml_plot`,
  `src/plugins/ml_plot`, `tests/plugins/ml_plot`

## Targets

- `leakflow_ml` (links Torch only; no `leakflow_core` dependency).
- `leakflow_plugins_ml` (wraps the numeric layer in pipeline elements/payloads).
- `leakflow_plugins_ml_plot` (introduced in A4 and extended by the implemented
  post-A4 bounded comparison view; links `leakflow_plugins_ml` and
  `leakflow_plot` and translates the evaluation payload into generic tabbed
  table and bounded heatmap `TableView` data only).

The pipeline **elements** that wrap these APIs live in the `leakflow_plugins_ml`
family (klass `Analyze/Clustering/...`, `Analyze/Evaluation/Clustering`,
`Transform/Feature/Select`). Generic `HeatmapPlot` lives in
`leakflow_plugins_plot`; the A4 clustering table bridge belongs in
`leakflow_plugins_ml_plot`, not either numeric target. Generic `MetricView` and
the standalone/selectable clustering matrix plot remain later work; the bounded
row-normalized Heatmap tab was added by the user-requested post-A4 extension to
`ClusteringMetricsTablePlot`.

Every payload-producing wrapper stamps the core `payload.layout` contract with
semantic axes after attaching its Torch payload: `FeatureSelect` emits
`observation/feature` or `unit/observation/feature`, `GaussianMixture` emits
`observation` or `unit/observation`, and `ClusteringStats` emits
`true_class/cluster` or `unit/true_class/cluster`.

The unit axis also carries identity as typed `Buffer::units()` (a `Units`):
`AesLeakage` (from `units`) and `CorrelationPoiToIndexes` (from `units`) set
it, `FeatureSelect` and `GaussianMixture` carry it through, and `ClusteringStats`
aligns its `labels` and `truth` inputs on it before scoring — **disjoint units are
an error**, a **partial overlap warns** and scores the shared units, and unlabeled
inputs fall back to a plain shape check. See `docs/design/metadata_klass_taxonomy.md`.

## Current API

- `gaussian_mixture.hpp`: `GaussianMixture` — GMM/EM modelled on
  `sklearn.mixture.GaussianMixture`, batched over a leading unit axis so the SCA
  `[U, T, N]` layout (U units, T traces, N features) fits one GMM per unit in one
  call. `covariance_type` full/diagonal; `kmeans++` / `random_from_data` / `random`
  init; `n_init` (best per unit); `reg_covar`/`tol`/`max_iter` (`max_iter=0` =
  init-only, for evaluating the init step); `fit`/`predict`/`predict_proba`/
  `score_samples` with `weights_/means_/covariances_/converged_/n_iter_/lower_bound_`.
  Also **size-constrained fitting** (`target_sizes` + `sinkhorn_epsilon`): plain EM
  warm start, sizes bound to components by mass rank, then a Sinkhorn-constrained
  E-step pins the assignment to the known cluster sizes — the GMM realisation of
  **Genevay et al. 2019**; `epsilon` dials soft↔hard. float64; cross-validated vs
  sklearn to ~1e-8 log-likelihood. Deferred: tied/spherical cov, warm_start, caller
  inits, `sample()`, `aic`/`bic`.
  Its optional progress callback can stop after an EM/constrained iteration; a
  separate cooperative checkpoint also runs between k-means++ initialization
  steps and before iterations. Cancellation returns the best partial numeric fit.
  The pipeline `GaussianMixture` wrapper binds those checkpoints to session Pause
  and discards the partial fit on Stop, so no canceled labels travel downstream.
  Its labels output also reports the fitted feature-axis width as
  `payload.cluster.n_features`.
- `sinkhorn.hpp`: `sinkhorn(cost, row_marginal, col_marginal, options)` — log-domain
  entropic-OT Sinkhorn (**Cuturi 2013**), batched over the unit axis; domain-agnostic.
  Cross-validated vs Python POT (`ot.sinkhorn`) to ~1e-13 on the plan.
- `clustering_metrics.hpp`: ground-truth clustering evaluation, batched over the unit
  axis. Label-free (permutation-invariant): `confusion_matrix`, `cluster_purity`,
  `adjusted_rand_index`, `normalized_mutual_info` (cross-validated vs
  `sklearn.metrics`). Matching-based (square C==K): `hungarian_match` (Kuhn-Munkres,
  cross-validated vs `scipy.optimize.linear_sum_assignment`), `reorder_confusion_columns`
  (diagonalise for plotting), and `matched_clustering_scores` (accuracy +
  per-class precision/recall/F1).
- `clustering_evaluation.hpp`: the core-free, GMM-independent
  `evaluate_clustering(...)` API through completed Phase A. It accepts arbitrary
  int64-representable predicted IDs `[N]`/`[U,N]` and numeric semantic truth
  `[N,D]`/`[U,N,D]`, groups truth by exact full-vector equality, and returns a
  structured per-unit result. Current results include deterministic sparse
  contingency detail, checked 64-bit unordered-pair counts, ARI, arithmetic
  AMI, homogeneity, completeness, V-measure, purity, pair
  precision/recall/F1, and arithmetic NMI. Metric records carry value,
  denominator support, descriptor metadata, and explicit undefined reasons.
  Optional power semantics add validated ranges/weights with `power=1|2`,
  semantic impurity micro/macro and per-dimension/per-cluster records,
  merge-error rate and conditional severity, plus fragmentation micro/macro and
  per-group records. Semantic records remain discoverable as unavailable in
  exact-only mode; fragmentation remains available. Optional rectangular
  exact-overlap and semantic-cost alignments are separate records with canonical
  mapping identities, deterministic predicted-major ties, unmatched marginal
  supports, exact per-group scores, semantic per-dimension errors, and
  Full-detail contingency-mass error records. Schema v5 appends
  `semantic_partition_separation = 1 - D_within / D_all` and the optional
  `semantic_partition_quality`, its harmonic mean with exact pair recall. The
  original `combined_quality` remains unchanged but deprecated because it is not
  comparable across predicted-cluster counts.
- `ClusteringEvaluate`: pipeline wrapper with `labels` and `truth` inputs and an
  `evaluation` output. It aligns typed unit identity carried by the input
  `Buffer`s, emits that identity on the output `Buffer`, and produces a bounded
  `ClusteringEvaluationPayload`. The payload stores effective `evaluation.*`
  options plus only bounded labels-side `payload.cluster.*` producer metadata;
  it does not own typed `Units` or copy `payload.parameter.*` metadata.
- `ClusteringMetricsTablePlot`: ML→plot bridge with a `sink` input. It translates
  structured results into generic `TableView` tab groups and adds direct
  evaluation-buffer `payload.parameter.*` metadata without recomputation. The
  original A4 bridge supplied seven table tabs; the user-requested bounded
  post-A4 extension supplies the explicit N/S columns and eighth Heatmap. Its
  tabs are Overview, Exact, Semantic, Fragmentation, Combined, Alignment,
  Heatmap, and Parameters. Overview uses one row per run and unit with explicit
  `Observations (N)` and `Features (S)` columns; the feature value comes from
  captured `labels.cluster.n_features` and is `N/A` when absent. It promotes
  semantic partition separation and the opt-in semantic partition quality; the
  deprecated combined quality remains only in Combined detail. Family tabs
  retain each stored `MetricValue` exactly once; Parameters records context once
  per run and retains the feature-count parameter. Heatmap uses only stored Full-detail
  sparse contingency, applies a stored exact-overlap column permutation when
  present (raw otherwise), row-normalizes copied counts, and labels canonical
  truth-vector rows and actual predicted-ID columns. Unit-bearing tabs share a
  view-local selector keyed by the typed unit ids on the evaluation `Buffer`.
  Heatmap supports ragged per-unit shapes, independent accumulated run frames,
  an unavailable Global-detail page, and a combined 1,000,000-cell per-run
  dense-display cap across unit pages.

The current pipeline `ClusteringStats` is intentionally narrower than the active
Phase A evaluator. It accepts scalar truth IDs, emits a dense reordered confusion
tensor, and attaches ARI/NMI/purity/matched scores as metadata. Its square
matching and tensor-output contract is retained for compatibility with existing
`ClusteringStats ! HeatmapPlot` graphs.

## Active Evaluation Sequence

Authoritative design: `docs/design/clustering_evaluation_metrics.md`.

### Phase A — Full Clustering Evaluation Metrics

- **A1 done:** GMM-independent `evaluate_clustering(...)`, structured
  `ClusteringEvaluationResult`, vector-truth exact grouping, sparse contingency,
  and the complete exact metric set.
- **A2 done:** semantic merge rate/severity, micro/macro/per-dimension and
  per-cluster impurity, and micro/macro/per-group fragmentation.
- **A3 done:** exact-overlap and semantic-cost rectangular alignments, including
  both rectangular directions, strict tie behavior, and fixed semantic dummy
  penalty semantics.
- **A4 done:** `ClusteringEvaluate`, `ClusteringEvaluationPayload`, typed-unit
  alignment, optional combined quality (now deprecated legacy data), bounded
  summaries, effective `evaluation.*` options, and bounded generic
  `payload.cluster.*` capture only from labels metadata in
  `leakflow_plugins_ml`; plus
  `ClusteringMetricsTablePlot` in `leakflow_plugins_ml_plot`. The table reuses
  generic domain-free `TableView` tabs and accepts explicitly stamped
  `payload.parameter.*` metadata on its input `Buffer`. Overview compares one
  run/unit per row using headline metrics and core producer/experiment
  parameters. Exact, Semantic, Fragmentation, Combined, and Alignment contain
  every stored `MetricValue` exactly once, with `↑`/`↓` direction labels;
  Parameters presents effective/captured context once per run.
  `update_mode` is
  `UiControl`/`ElementUi`, accepts `auto|accumulate|replace`, and publishes its
  read-only resolved `active_update_mode`; auto follows liveness. Group/title
  and view-local unit/tab/clear/sort controls are `UiControl`. Accumulation
  retains comparison rows and run history. It does not introspect arbitrary upstream properties,
  flatten the structured payload into metadata, or recompute metrics.
- **Post-A4 bounded comparison-view extension done (user-requested):** Overview
  exposes `Observations (N)` and producer-supplied `Features (S)`, with GMM
  reporting S as `payload.cluster.n_features`. An eighth same-window Heatmap tab
  displays stored Full-detail contingency as a bounded row-normalized matrix,
  exact-aligned when the stored mapping is available, with shared typed-unit and
  independent run-history controls. It adds no persistence, standalone plot
  element, selectable matrix mode, or recomputation.
- **Post-A4 clustering-quality correction done (user-requested):** schema v5
  appends semantic partition separation and default-off semantic partition
  quality. Separation is `1 - D_within / D_all`; quality is its harmonic mean
  with exact pair recall. Perfect/collapse/singleton endpoints are `1/0/0`, and
  zero total semantic variation is explicit unavailability. The legacy
  `combined_quality` option and record are preserved unchanged but deprecated;
  Overview uses the corrected values and Combined keeps legacy detail.
- Support exact-only evaluation without ranges. Semantic mode uses an explicit
  normalized power cost with configured ranges/weights and `power=1|2` (default
  2). Keep undefined value, reason, support, direction, family, and averaging in
  every metric record.
- Do not mutate `ClusteringStats`; it remains the legacy matrix adapter.

### Deferred Follow-up (Unblocked)

Completed A4 and its bounded post-A4 extension leave versioned payload
persistence and round-trip coverage unblocked but deferred. A later visual
slice may add generic
`MetricView`/`ClusteringMetricsPlot` and
the standalone `ClusteringMatrixPlot` with selectable raw/exact/semantic matrix
and normalization modes; those bridge elements consume stored payload results
and apply display transformations only. They do not compute metrics,
assignments, or clustering labels.

### Validation

- A1 conventional metrics are cross-validated against checked-in scikit-learn
  fixtures, with degeneracy, symmetry, arbitrary-ID, vector/batch/dtype,
  validation, sparse-detail, and repeated-marginal AMI stress coverage.
- A2 semantic and fragmentation metrics are checked against `p=1`/`p=2`
  hand fixtures, batch-local ranges, power-mode numeric dtypes, zero weights,
  undefined denominators, and non-quadratic stress cases.
- A3 alignments are checked with hand-computed `G<K` and `G>K` fixtures,
  mappings that deliberately differ by objective, both powers, `D=1/2/4`,
  batches, zero weights, dtype edges, strict ties, and large-N/small-assignment
  stress.
- The A4 legacy combined-score case is hand-checked. Schema-v5 correction tests
  hand-check `p=1` and `p=2`, perfect/collapse/singleton endpoints,
  no-semantic-variation unavailability, and option validation.
- Cover arbitrary IDs, rectangular mappings, `D=1/2/4`, unit batching/alignment,
  all undefined denominators, invalid ranges/weights/power, overflow, and a
  non-quadratic stress case.
- A4 coverage includes payload summary/effective options and bounded
  producer-parameter capture, plus tab translation, one-row-per-run/unit
  overview, every-value-once family coverage, parameters once per run, explicit
  `payload.parameter.*` metadata, accumulate/replace history, typed-unit
  selection, clear, per-tab column sorting, and no-recomputation behavior.
- Post-A4 extension coverage includes GMM feature-width capture, explicit N/S
  comparison columns, ragged stored-contingency Heatmaps, unavailable/cell-limit
  behavior, and independent accumulated Heatmap frames.
- Preserve all current `ClusteringStats` tests and pipeline behavior.

## Contracts

- `leakflow_ml` has no dependency on `leakflow_core`, `leakflow_base`, or SCA
  layers; it is pure Torch/numeric code. `leakflow_plugins_ml` supplies the
  pipeline/core integration.
- All heavy math runs in float64 for stability.
- Structured evaluation details belong in the ML plugin payload, not string
  metadata. The numeric result itself remains core-free.
- Metric IDs are append-only. Schema v5 appends semantic partition separation,
  corrected quality, and `no_semantic_variation` without renumbering legacy
  records.
- The A4 plot dependency is isolated in `leakflow_plugins_ml_plot`; neither
  `leakflow_ml` nor `leakflow_plugins_ml` links plotting.
- Payload persistence, `MetricView`, and the standalone/selectable clustering
  matrix plot are not part of A4 or the bounded post-A4 extension. That
  extension's fixed same-window Heatmap tab is implemented.

## Common Tests

```bash
ctest --test-dir build -R 'leakflow_ml|leakflow_plugins_ml'
```
