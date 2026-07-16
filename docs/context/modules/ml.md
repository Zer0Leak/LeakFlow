# ML Module Context

Use this for the generic (domain-free) machine-learning / statistics numeric layer:
clustering, mixture models, optimal transport, and clustering evaluation. No SCA or
AES knowledge lives here.

## Files

- Numeric public headers: `include/leakflow/ml`
- Numeric sources/tests: `src/ml`, `tests/ml`
- Pipeline public headers: `include/leakflow/plugins/ml`
- Pipeline sources/tests: `src/plugins/ml`, `tests/plugins/ml`

## Targets

- `leakflow_ml` (links Torch only; no `leakflow_core` dependency).
- `leakflow_plugins_ml` (wraps the numeric layer in pipeline elements/payloads).

The pipeline **elements** that wrap these APIs live in the `leakflow_plugins_ml`
family (klass `Analyze/Clustering/...`, `Analyze/Evaluation/Clustering`,
`Transform/Feature/Select`). Generic `HeatmapPlot` lives in
`leakflow_plugins_plot`; a future clustering-result plot bridge belongs in
`leakflow_plugins_ml_plot`, not either numeric target.

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

The current pipeline `ClusteringStats` is intentionally narrower than the next
planned evaluator. It accepts scalar truth IDs, emits a dense reordered confusion
tensor, and attaches ARI/NMI/purity/matched scores as metadata. Its square
matching and tensor-output contract is retained for compatibility with existing
`ClusteringStats ! HeatmapPlot` graphs.

## Planned Evaluation Phases (Not Implemented)

Authoritative design: `docs/design/clustering_evaluation_metrics.md`.

### Phase A — Full Clustering Evaluation Metrics

- Add a GMM-independent `evaluate_clustering(...)` API and structured
  `ClusteringEvaluationResult` in `leakflow_ml`.
- Accept labels `[N]`/`[U,N]` and vector semantic truth
  `[N,D]`/`[U,N,D]`; derive exact groups from full-vector equality.
- Add ARI, arithmetic AMI, homogeneity, completeness, V-measure, pair
  precision/recall/F1, purity, compatibility NMI, semantic merge rate/severity and
  micro/macro/per-dimension impurity, micro/macro/per-group fragmentation, plus
  requested exact-overlap and semantic-cost rectangular alignments.
- Support exact-only evaluation without ranges. Semantic mode uses an explicit
  normalized power cost with configured ranges/weights and `power=1|2` (default
  2). Keep undefined value, reason, support, direction, family, and averaging in
  every metric record.
- Add `ClusteringEvaluate` and `ClusteringEvaluationPayload` in
  `leakflow_plugins_ml`, including typed-unit alignment, bounded summaries, and
  versioned persistence.
- Do not mutate `ClusteringStats`; it remains the legacy matrix adapter.

### Phase B — Metric Visualization

After Phase A freezes the payload contract, add `leakflow_plugins_ml_plot`.
Bridge elements consume the payload and fill domain-free table/metric/heatmap
views. They select stored results and apply display normalization only; they do
not compute metrics, assignments, or clustering labels.

### Planned Validation

- Cross-validate conventional metrics against checked-in scikit-learn fixtures.
- Hand-check semantic, pair, fragmentation, combined-score, and alignment cases.
- Cover arbitrary IDs, rectangular mappings, `D=1/2/4`, unit batching/alignment,
  all undefined denominators, invalid ranges/weights/power, overflow, and a
  non-quadratic stress case.
- Preserve all current `ClusteringStats` tests and pipeline behavior.

## Contracts

- `leakflow_ml` has no dependency on `leakflow_core`, `leakflow_base`, or SCA
  layers; it is pure Torch/numeric code. `leakflow_plugins_ml` supplies the
  pipeline/core integration.
- All heavy math runs in float64 for stability.
- Structured evaluation details belong in the ML plugin payload, not string
  metadata. The numeric result itself remains core-free.

## Common Tests

```bash
ctest --test-dir build -R leakflow_ml
```
