# ML Module Context

Use this for the generic (domain-free) machine-learning / statistics numeric layer:
clustering, mixture models, optimal transport, and clustering evaluation. No SCA or
AES knowledge lives here.

## Files

- Public headers: `include/leakflow/ml`
- Sources: `src/ml`
- Tests: `tests/ml`

## Target

- `leakflow_ml` (links Torch only; no `leakflow_core` dependency).

The pipeline **elements** that wrap these APIs live in the `leakflow_plugins_ml`
family (klass `Analyze/Clustering/...`, `Analyze/Evaluation/Clustering`,
`Transform/Feature/Select`, `Plot/Heatmap`).

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

## Contracts

- No dependency on `leakflow_core`, `leakflow_base`, or SCA layers. Pure Torch numeric.
- All heavy math runs in float64 for stability.

## Common Tests

```bash
ctest --test-dir build -R leakflow_ml
```
