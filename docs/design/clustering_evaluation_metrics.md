# Clustering Evaluation and Metric Visualization

Status: **Phase A implemented**. A1 (exact numeric core), A2 (semantic and
fragmentation metrics), A3 (rectangular alignments), and A4 (pipeline result
contract plus table-only inspection) are complete. Payload persistence, generic
`MetricView`, and matrix plotting are unblocked but remain deferred.

This design adds a complete, clustering-algorithm-independent evaluation layer
and a follow-up visualization layer. The motivating experiments use GMM labels
and Hamming-weight semantics, but no API or implementation may depend on GMM,
AES, Kyber, a fixed number of semantic dimensions, or a fixed number of classes.

## Phase Split

Implementation is deliberately split into bounded slices.

Implemented Phase A slices:

- **A1 — exact numeric core (implemented):** core-free result/options contract,
  vector-truth grouping, deterministic sparse contingency detail, conventional
  exact metrics, explicit pair-score undefined/support semantics, and numeric
  tests.
- **A2 — semantic and fragmentation metrics (implemented):** normalized
  power-cost validation and aggregate/detail metrics. Supplied semantic
  configuration is validated even in `off` mode but ranges are not required;
  semantic records then carry `semantic_disabled`. Full detail retains
  singleton cluster/group records with explicit undefined reasons.
- **A3 — rectangular alignments (implemented):** exact-overlap and semantic-cost
  assignments with explicit unmatched marginal support, strict predicted-major
  dense tie-breaking, exact per-group scores, semantic per-dimension errors,
  and Full-detail contingency-mass error records.
- **A4 — pipeline and table-inspection contract (implemented):** the default-off
  optional combined-quality record/property, payload, evaluator element,
  typed-unit alignment, bounded summary and parameter capture, registration,
  and pipeline tests. It also adds `ClusteringMetricsTablePlot` in a new
  `leakflow_plugins_ml_plot` bridge, reusing generic `TableView` for metrics and
  captured parameters with `replace|append`, clear, and deterministic
  per-column `asc|desc` sorting.

### Phase A — Full Clustering Evaluation Metrics

Phase A owns:

- generic numeric evaluation in `leakflow_ml`;
- a structured result contract;
- a new pipeline evaluator and payload in `leakflow_plugins_ml`;
- exact, semantic-aware, and alignment metrics;
- bounded summary and effective-option/parameter capture for the structured
  payload;
- the table-only `leakflow_plugins_ml_plot` bridge and any domain-free sorting
  support required in the existing `TableView`;
- compatibility with the existing `ClusteringStats` element;
- numeric, pipeline, and table-bridge tests.

A4 does not add payload persistence, a generic `MetricView`, or matrix plotting.
The plot dependency is isolated in `leakflow_plugins_ml_plot`; neither
`leakflow_ml` nor `leakflow_plugins_ml` links plotting.

### Deferred Follow-up — Persistence and Additional Visualization (Unblocked)

The deferred follow-up owns:

- versioned payload persistence and round-trip compatibility;
- domain-free metric display data/views in `leakflow_plot`;
- additional clustering-result-to-view elements in the A4-created
  `leakflow_plugins_ml_plot` target;
- normalized metric charts and result-matrix heatmaps;
- headless view-data tests and manual GUI validation.

The follow-up can consume the now-stable A4 results. It must not recompute
clustering metrics or alignment assignments.

## Goals

- Compare arbitrary predicted cluster identifiers with vector-valued semantic
  truth.
- Keep exact partition recovery separate from semantic error severity.
- Report merge and fragmentation behavior independently.
- Support both one dataset and a leading unit batch.
- Represent undefined results explicitly.
- Avoid observation-pair enumeration.
- Make every result self-describing enough for bounded summaries and table
  inspection; keep a clean seam for later persistence and additional plots.

## Non-Goals

- Changing how `GaussianMixture` fits or predicts.
- Feeding semantic truth into clustering.
- Defining a universal scalar that replaces the complete metric set.
- Encoding or decoding AES-specific class identifiers.
- Turning `leakflow_ml` into an SCA-specific library.
- Adding payload persistence, a generic metric chart, or a clustering matrix
  plot during A4.

## 1. Input Contract

For each unit, the evaluator receives \(N\) observations.

Predicted cluster identifiers:

\[
c_i,\qquad i=1,\ldots,N
\]

Ground-truth semantic vectors:

\[
\mathbf h_i=(h_{i1},h_{i2},\ldots,h_{iD}),\qquad D\geq1
\]

Two observations belong to the same exact truth group if and only if their full
semantic vectors are equal:

\[
\mathbf h_i=\mathbf h_j.
\]

The first implementation accepts:

- predicted labels with shape `[N]` or `[U,N]`;
- semantic truth with shape `[N,D]` or `[U,N,D]`;
- arbitrary integral predicted identifiers, including negative and sparse IDs;
- finite discrete numeric semantic coordinates for which exact equality is
  meaningful;
- optional semantic dimension names, otherwise deterministic names
  `dimension_0`, `dimension_1`, and so on;
- typed unit identity on pipeline buffers.

The numeric API normalizes arbitrary identifiers and distinct semantic vectors
to dense internal IDs. It must not require callers to encode semantic vectors
into scalar class IDs.

For batched input, all units must have the same observation count and semantic
dimension count. The pipeline element applies the existing typed-unit alignment
contract before numeric evaluation: disjoint units are an error, partial
overlap warns and evaluates shared units, and unlabeled buffers fall back to
shape validation.

Invalid input is an error, including empty input, non-finite semantic values,
shape mismatch, or a semantic dimension count inconsistent with configuration.

## 2. Semantic Power Cost

Standard clustering metrics treat all wrong truth groups as categorically
different. A separate cost measures how severe a semantic mismatch is.

For dimension \(d\), configure:

- \(R_d>0\): the largest meaningful coordinate difference;
- \(w_d\geq0\): a dimension weight;
- \(p\in\{1,2\}\): the power exponent.

At least one weight must be positive. The normalized semantic power cost is:

\[
s_p(\mathbf a,\mathbf b)=
\frac{\sum_{d=1}^{D}w_d
\left(\frac{|a_d-b_d|}{R_d}\right)^p}
{\sum_{d=1}^{D}w_d}.
\]

Semantic evaluation is explicit. In `semantic=off` mode, the evaluator still
computes exact partition and fragmentation metrics and does not require ranges.
In `semantic=power` mode, the default is \(p=2\) with equal weights. Ranges are
required because silently inferring a range from one dataset would make results
incomparable across runs.

This quantity is intentionally named a **power cost**, not a distance. Without
the outer \(p\)-th root, it is not generally a mathematical metric for \(p>1\).
The no-root form is retained because squared errors deliberately penalize large,
unbalanced semantic mistakes.

Phase A deliberately bounds \(p\) to 1 or 2. Those cases have exact aggregated
algorithms that do not enumerate observation pairs. Arbitrary real powers are
deferred rather than weakening the no-\(O(N^2)\) contract.

Phase A uses strict range validation:

\[
\max_i h_{id}-\min_i h_{id}\leq R_d
\]

for every evaluated unit and dimension. Out-of-range values are rejected; they
are never silently clipped. With valid ranges:

\[
0\leq s_p(\mathbf a,\mathbf b)\leq1.
\]

For two byte Hamming-weight coordinates with ranges `[8,8]`, equal weights,
and \(p=2\):

\[
s_2(\mathbf a,\mathbf b)=
\frac{\Delta HW_1^2+\Delta HW_2^2}{128}.
\]

## 3. Exact Partition Metrics

Let \(g_i\) be the dense truth-group ID derived from \(\mathbf h_i\). Define the
contingency counts:

\[
n_{gk}=\#\{i:g_i=g,\ c_i=k\},
\]

with truth groups as rows and predicted clusters as columns.

Phase A reports:

- Adjusted Rand Index (ARI), the primary conventional score;
- Adjusted Mutual Information (AMI), using arithmetic normalization;
- homogeneity;
- completeness;
- V-measure with \(\beta=1\);
- cluster purity;
- pair precision;
- pair recall;
- pair F1;
- compatibility NMI using arithmetic normalization.

Purity and NMI are retained for compatibility with current `ClusteringStats`.
Purity must be interpreted with fragmentation because it can be maximized by
over-splitting. NMI is not a second primary score because arithmetic NMI and
V-measure with \(\beta=1\) are equivalent.

Cluster purity is:

\[
\operatorname{purity}=\frac{1}{N}\sum_k\max_g n_{gk}.
\]

ARI, AMI, homogeneity, completeness, V-measure, and NMI follow the documented
scikit-learn conventions, including degenerate partitions. The implementation
must be cross-validated against scikit-learn fixtures rather than inferred from
one example.

### Pair Counts

For unordered pairs \(i<j\):

\[
\begin{aligned}
TP &: c_i=c_j\ \land\ g_i=g_j,\\
FP &: c_i=c_j\ \land\ g_i\ne g_j,\\
FN &: c_i\ne c_j\ \land\ g_i=g_j.
\end{aligned}
\]

Calculate:

\[
P_{\mathrm{pair}}=\frac{TP}{TP+FP},\qquad
R_{\mathrm{pair}}=\frac{TP}{TP+FN},
\]

\[
F1_{\mathrm{pair}}=
\frac{2P_{\mathrm{pair}}R_{\mathrm{pair}}}
{P_{\mathrm{pair}}+R_{\mathrm{pair}}}.
\]

Pair precision is undefined when there are no predicted within-cluster pairs.
Pair recall is undefined when there are no true within-group pairs. Pair F1 is
undefined if either input is undefined; when both are defined and both are zero,
F1 is zero.

## 4. Semantic Merge Metrics

Let \(C_k\) be predicted cluster \(k\), with size \(n_k\).

For each non-singleton predicted cluster:

\[
I_k=
\frac{\sum_{i<j,\ i,j\in C_k}s_p(\mathbf h_i,\mathbf h_j)}
{\binom{n_k}{2}}.
\]

Report the following.

### Micro Semantic Impurity

\[
I_{\mathrm{micro}}=
\frac{\sum_k\sum_{i<j,\ i,j\in C_k}s_p(\mathbf h_i,\mathbf h_j)}
{\sum_k\binom{n_k}{2}}.
\]

Every predicted within-cluster pair has equal weight.

### Macro Semantic Impurity

\[
I_{\mathrm{macro}}=
\frac{1}{|\mathcal C_2|}
\sum_{k\in\mathcal C_2}I_k,
\]

where \(\mathcal C_2\) is the set of non-singleton predicted clusters. Every
eligible predicted cluster has equal weight.

### Merge-Error Rate

\[
M=\frac{FP}{TP+FP}=1-P_{\mathrm{pair}}.
\]

### Conditional Merge-Error Severity

\[
S_{\mathrm{merge}}=
\frac{\sum_{i<j,\ c_i=c_j,\ g_i\ne g_j}
s_p(\mathbf h_i,\mathbf h_j)}
{FP}.
\]

When merge errors exist:

\[
I_{\mathrm{micro}}=M\,S_{\mathrm{merge}}.
\]

If predicted within-cluster pairs exist but \(FP=0\), merge rate and impurity
are zero while conditional severity is undefined. This distinction must be
preserved.

### Per-Dimension Impurity

For each dimension, report micro and macro forms using:

\[
s_{p,d}(a,b)=
\left(\frac{|a_d-b_d|}{R_d}\right)^p.
\]

Per-dimension metrics are not multiplied by \(w_d\); the configured weight is
reported separately. This lets users see the raw normalized behavior of a
dimension even when it has zero weight in the aggregate cost.

## 5. Fragmentation Metrics

Merge metrics measure incompatible observations placed together.
Fragmentation measures compatible observations split apart.

### Micro Fragmentation

\[
F_{\mathrm{micro}}=
\frac{FN}{TP+FN}=1-R_{\mathrm{pair}}.
\]

### Per-Truth-Group Fragmentation

For truth group \(g\), with size \(n_g\):

\[
F_g=
1-\frac{\sum_k\binom{n_{gk}}{2}}{\binom{n_g}{2}},
\qquad n_g\geq2.
\]

### Macro Fragmentation

\[
F_{\mathrm{macro}}=
\frac{1}{|\mathcal G_2|}
\sum_{g\in\mathcal G_2}F_g,
\]

where \(\mathcal G_2\) is the set of non-singleton truth groups.

No semantic power cost is applied to a fragmented pair: equal semantic vectors
have zero semantic cost, while the error is their separation.

## 6. Optional Combined Quality

Implementation scheduling: this optional record and its default-off property
were added in A4 with the pipeline payload/property contract. A2 reports both
component metrics but does not calculate the composite.

A composite score may be requested for ranking experiments, but it is never the
only reported result.

Define semantic cohesion and group preservation:

\[
C_{\mathrm{semantic}}=1-I_{\mathrm{micro}},\qquad
G_{\mathrm{preservation}}=1-F_{\mathrm{micro}}
=R_{\mathrm{pair}}.
\]

Then:

\[
Q=
\frac{2C_{\mathrm{semantic}}G_{\mathrm{preservation}}}
{C_{\mathrm{semantic}}+G_{\mathrm{preservation}}}.
\]

`combined_quality` is disabled by default. If requested, it is undefined when
either underlying metric is undefined and is always accompanied by both
underlying values. If both defined components are zero, the harmonic score is
defined as zero.

## 7. Alignment Results

Core metrics are permutation invariant and do not need an assignment between
clusters and truth groups. Alignment is optional post-processing for
interpretation and visualization only.

Both alignment methods support rectangular \(G\times K\) inputs by padding the
assignment problem to \(L=\max(G,K)\). Solver rows are canonical predicted
clusters followed by dummy predicted rows; columns are canonical truth groups
followed by dummy truth columns. Mappings use `-1` for unmatched entries.

Tie-breaking is part of the numeric contract, not an implementation accident:
after optimizing the primary objective, choose the lexicographically smallest
predicted-cluster-to-truth-group vector in ascending canonical predicted-cluster
order, treating unmatched as larger than every real truth-group index. Thus real
truth groups precede dummies and dense truth-group order breaks remaining ties.
The implementation refines the Hungarian equality graph, so the secondary rule
does not perturb or weaken the primary optimum.

Each alignment reports four separate marginal supports: assigned and unmatched
predicted-cluster observations, and assigned and unmatched truth-group
observations. These are deliberately not collapsed into one ambiguous
"matched support" field. A rectangular assignment may pair real groups and
clusters with zero overlap because the maximum-cardinality real-to-real pairing
is still explicit.

### Exact-Overlap Alignment

Maximize:

\[
\sum_{(g,k)\in\mathcal M}n_{gk}.
\]

Report:

- predicted-cluster-to-truth-group mapping;
- inverse truth-group-to-predicted-cluster mapping;
- unmatched canonical indices and their observation support;
- matched-overlap observation count, nonmatched count, and matched accuracy;
- a column permutation for the rectangular contingency matrix;
- per-truth-group precision, recall, F1, Jaccard, and support.

For truth group \(g\) assigned predicted cluster \(k\), let
\(t=n_{gk}\). The per-group records are:

\[
P_g=\frac{t}{n_k},\qquad
R_g=\frac{t}{n_g},\qquad
F1_g=\frac{2t}{n_g+n_k},\qquad
J_g=\frac{t}{n_g+n_k-t}.
\]

Their denominator supports are \(n_k\), \(n_g\), \(n_g+n_k\), and
\(n_g+n_k-t\), respectively. For an unmatched truth group, precision is
undefined with `no_matched_predicted_cluster`, recall and Jaccard are defined as
zero with support \(n_g\), and F1 is unavailable with
`dependent_metric_undefined`.

The aligned-column permutation means "new column position to original canonical
predicted-cluster index": matched clusters appear in assigned truth-group order,
then unmatched predicted clusters in ascending canonical order. The matrix
remains rectangular; no row or column is discarded and no dummy column is
materialized in the result.

### Semantic-Cost Alignment

For truth-group representative vector \(\mathbf v_g\), define:

\[
D_{kg}=
\sum_{g'}n_{g'k}s_p(\mathbf v_{g'},\mathbf v_g).
\]

Minimize total assigned cost. A predicted cluster assigned to a dummy truth
group receives its maximum normalized penalty \(n_k\). Dummy predicted rows have
zero cost but remain explicitly unmatched. The normalized dummy penalty is
fixed at `1.0` and is stored explicitly in the result. For an unmatched
predicted cluster the aggregate error and every per-dimension error are `1.0`,
preserving the configured weighted identity.

If \(a(k)\) is the assigned truth group for predicted cluster \(k\), define for
a real assignment:

\[
e_{g'k,d}=\left(\frac{|v_{g',d}-v_{a(k),d}|}{R_d}\right)^p,
\qquad
e_{g'k}=\frac{\sum_d w_d e_{g'k,d}}{\sum_d w_d}.
\]

For an unmatched predicted cluster, \(e_{g'k,d}=e_{g'k}=1\). Report:

\[
E=\frac{1}{N}\sum_{g',k}n_{g'k}e_{g'k},
\qquad
E_d=\frac{1}{N}\sum_{g',k}n_{g'k}e_{g'k,d}.
\]

Therefore:

\[
E=\frac{\sum_d w_dE_d}{\sum_d w_d}.
\]

Report:

- the semantic-cost mapping, inverse mapping, four marginal supports, and
  unmatched canonical-index records;
- exact-overlap and nonoverlap observation counts under the semantic mapping;
- penalized normalized cost \(E\) and per-dimension errors \(E_d\), each with
  support \(N\);
- in Full detail, one deterministic error-mass record per nonzero contingency
  cell: source truth index, predicted index, assigned truth index or unmatched,
  observation count, aggregate cost, and per-dimension costs.

The error-mass records are ordered by the canonical sparse contingency order and
sum to \(N\); they describe the empirical error distribution without expanding
observations or merging floating values into unstable bins. Global detail keeps
the mapping, identities, supports, and scalar/per-dimension metrics but omits
per-truth exact scores and semantic error masses.

Exact-overlap and semantic-cost mappings must never share one ambiguous
`alignment` field because they optimize different objectives.

## 8. Result Contract

The numeric layer introduces a plain `ClusteringEvaluationResult` in
`leakflow_ml`. It has no `leakflow_core` dependency.

Every metric value carries:

- stable metric name;
- optional numeric value;
- family: exact, semantic, fragmentation, or alignment;
- direction: higher-is-better or lower-is-better;
- averaging: none, micro, macro, per-cluster, per-group, or per-dimension;
- support counts used by its denominator;
- an undefined reason when no value exists.

In exact-only mode, semantic metric records remain discoverable but unavailable
with reason `semantic_disabled`; they are not silently omitted. Requesting
semantic-cost alignment while semantic evaluation is disabled is a
configuration error.

The result contains:

- schema version and effective options;
- semantic dimension names, ranges, weights, and power;
- one result record per aligned unit; the enclosing pipeline `Buffer` carries
  the corresponding typed unit identities;
- observation, truth-group, predicted-cluster, and eligible-pair counts;
- all required global metrics;
- per-dimension metrics;
- optional per-cluster and per-group details;
- optional contingency and alignment records.

When either alignment is requested, canonical predicted IDs and truth vectors
are materialized as alignment identities even in Global detail, so dense mapping
indices remain self-describing. Full partition detail additionally retains the
sparse contingency. Exact alignment can be requested with `semantic=off`;
semantic alignment, alone or together with exact alignment, requires
`semantic=power` and validated ranges/weights.

Large or structured results are not encoded as string metadata.

The pipeline layer adds `ClusteringEvaluationPayload` in
`leakflow_plugins_ml`. It wraps the numeric result, derives from `Payload`,
uses caps `leakflow/clustering-evaluation`, and declares the structured layout
`unit/evaluation`. The enclosing output `Buffer` retains typed `Units`; unit
identity is not duplicated inside the payload. The payload records the effective
evaluator options used to produce the result as bounded `evaluation.*`
parameters.

The evaluator also captures a bounded set of generic clustering-producer
parameters only from the labels buffer metadata namespace `payload.cluster.*`
(for example `method`, `n_components`, `covariance_type`, and `converged`). Captured
records are scalar, deterministically key-ordered, and subject to explicit
entry/key/value bounds in the public payload contract. Unknown metadata
namespaces are ignored. The evaluator does not inspect arbitrary upstream
element properties, copy `payload.parameter.*` metadata, or flatten the
structured evaluation result into metadata.

The payload summary prints a bounded headline view: counts, ARI, AMI,
V-measure, purity, pair precision/recall/F1, semantic impurity, and fragmentation.
Undefined values render as `N/A` with a short reason. Full detail belongs in
the payload or table view, not unbounded terminal metadata.

`ClusteringEvaluationPayload` persistence is explicitly outside A4. A later
slice may add a versioned codec and `BufferFileSink`/`BufferFileSrc` round-trip
compatibility without changing A4's in-memory schema semantics.

## 9. Numeric and Pipeline APIs

The numeric API has one primary evaluator:

`evaluate_clustering(predicted, semantic_truth, options)`

It returns `ClusteringEvaluationResult` and does not know about buffers,
properties, plots, AES, or GMM.

The pipeline API provides `ClusteringEvaluate`:

- `labels` sink: predicted cluster IDs;
- `truth` sink: semantic vectors;
- `evaluation` source: `ClusteringEvaluationPayload`.

Properties:

- `semantic` = `off|power` (default `off`);
- `semantic_ranges` (required when `semantic=power`);
- `semantic_weights` (optional; equal weights by default);
- `power` = `1|2` (default `2`);
- `dimension_names` (optional);
- `detail` = `global|full`;
- `alignment` = `none|exact|semantic|both`;
- `combined_quality` (default `false`).

Metric-affecting properties are `payload-output` with downstream invalidation.
Display selection is not an evaluator property.

A4 also added `ClusteringMetricsTablePlot` with one `sink` input accepting
`ClusteringEvaluationPayload`. Its table contract is:

- show metric value/defined state, direction, family, averaging, support, and
  undefined reason, plus typed-unit/count context;
- show effective `evaluation.*` options and bounded labels-side
  `payload.cluster.*` parameters stored by the payload;
- accept scalar `payload.parameter.*` metadata explicitly stamped on the
  evaluation input as additional display parameters, while ignoring all other
  namespaces;
- expose payload parameters as `parameter.payload.<name>` and direct input
  metadata as `parameter.metadata.<name>`, so equal suffixes remain distinct;
- `update_mode=replace|append` (default `replace`), where replace refreshes the
  current comparison table and append retains it and adds the new translated
  result;
- provide an explicit clear operation that empties retained table state, also
  honored by `PlotRuntime::clear()`;
- allow any column to be selected as a stable sort key with `asc|desc` ordering.
  Numeric values sort numerically, text values lexically, unavailable values
  follow defined values in both directions, and ties retain source order.

`update_mode` is `SinkDisplay`/`ElementUi`: an Idle change re-derives the table
from the cached sink input without upstream work. `group` and `title` are
`UiControl`/`ElementUi`. Clear and table sorting are view-local UI controls.
None of these operations reruns clustering, evaluation, or alignment.

## 10. Compatibility and Migration

The current `ClusteringStats` contract remains available during migration:

- scalar truth class IDs;
- dense reordered confusion output;
- ARI, NMI, purity, and square-Hungarian matched scores in metadata.

Phase A must not silently change its accepted truth shape, output payload type,
or matrix layout. Existing `ClusteringStats ! HeatmapPlot` pipelines continue
to work.

`ClusteringEvaluate` is a separate element because its vector-truth input and
structured output are materially different contracts. Existing metric kernels
may be reused internally. `ClusteringStats` remains the legacy matrix adapter;
removal is outside A4 and its deferred visualization/persistence follow-up.

## 11. Table Inspection and Deferred Visualization Contract

### A4 Table Bridge

A4 added `leakflow_plugins_ml_plot`, depending on `leakflow_plugins_ml` and
`leakflow_plot`. It contains only `ClusteringMetricsTablePlot`. Clustering
semantics stay in the bridge and do not enter `leakflow_plot`.

The bridge reuses the existing domain-free `TableView`; A4 extended that
generic view only with reusable stable column sorting and clear/update behavior.
It does not add another plot-view type. The table translates the payload into:

- exact, semantic, fragmentation, alignment, per-dimension, per-cluster, and
  per-group metric rows as allowed by stored detail;
- value, direction, averaging, support, and `N/A` reason fields;
- typed-unit identity from the enclosing `Buffer` and
  observation/group/cluster count context;
- effective evaluator options;
- bounded labels-side producer parameters captured under `payload.cluster.*`
  and rendered as `parameter.payload.<name>`; and
- explicit evaluation-buffer `payload.parameter.*` metadata rendered as
  `parameter.metadata.<name>` for presentation.

`replace` and `append` have the deterministic semantics defined in Section 9.
Clear drops retained table content. Any column can be selected as the stable
sort key with `asc` or `desc`; sorting never separates cells that belong to one
row. Missing optional detail yields an unavailable row/section and never
triggers numeric recomputation.

The bridge only copies and translates stored data. A fixture-payload test proves
that it never calls `evaluate_clustering` or an assignment solver. Generic
`TableView` and `PlotRuntime` contain no clustering-field branches.

### Deferred Persistence and Views

The following are explicitly deferred beyond A4:

- a versioned `ClusteringEvaluationPayload` persistence codec;
- a domain-free `MetricView` and `ClusteringMetricsPlot` for grouped scalar and
  per-dimension bars; and
- `ClusteringMatrixPlot`, reusing `HeatmapView` for raw contingency,
  exact-overlap aligned, or semantic-cost aligned stored results with
  presentation-only `none|row|col` normalization.

When added, those views must consume stored payload results and preserve
unmatched/undefined state. Display selection and normalization must not rerun
clustering, evaluation, or alignment.

## 12. Computational Requirements

Observation pairs must not be explicitly materialized.

Pair counts use:

\[
TP=\sum_{g,k}\binom{n_{gk}}{2},\qquad
TP+FP=\sum_k\binom{n_k}{2},\qquad
TP+FN=\sum_g\binom{n_g}{2}.
\]

For \(p=2\), per-cluster/per-dimension semantic sums use:

\[
\sum_{i<j}(x_i-x_j)^2=
n\sum_i x_i^2-\left(\sum_i x_i\right)^2.
\]

For \(p=1\), sort each cluster/dimension and use prefix sums to obtain the total
absolute pair difference in \(O(n\log n)\). Semantic-cost alignment uses
contingency counts and truth-group representative vectors. It precomputes the
\(G\times G\) truth-group power costs, accumulates the \(K\times G\) assignment
matrix from the \(Z\) nonzero contingency cells, and never enumerates
observation pairs.

Use sparse/hash mappings for semantic vectors and contingency entries. A dense
matrix is materialized only when an alignment requests the assignment problem.
For \(L=\max(G,K)\), Hungarian assignment plus strict equality-graph tie
refinement is \(O(L^3)\) time and \(O(L^2)\) memory. Semantic matrix construction
is \(O(G^2D+ZG)\).

Use 64-bit counts with checked \(\binom n2\) arithmetic. Heavy floating-point
math uses at least float64 with extended/compensated accumulation where
available. Exact assignment costs use wider internal arithmetic so primary
overlap optimality is not perturbed by tie-breaking. AMI expected-mutual-
information terms require a stable log-gamma/hypergeometric implementation and
independent reference validation.

## 13. Implemented File Ownership

Phase A uses focused evaluation/bridge files rather than turning the current
compatibility header into an unrelated catch-all:

- `include/leakflow/ml/clustering_evaluation.hpp`
- `src/ml/clustering_evaluation.cpp`
- `tests/ml/clustering_evaluation_test.cpp`
- `include/leakflow/plugins/ml/clustering_evaluation_payload.hpp`
- `include/leakflow/plugins/ml/clustering_evaluate.hpp`
- `src/plugins/ml/clustering_evaluation_payload.cpp`,
  `src/plugins/ml/clustering_evaluate.cpp`,
  `src/plugins/ml/descriptor_catalog.cpp`, `src/plugins/ml/CMakeLists.txt`, and
  `tests/plugins/ml/{clustering_evaluate_test.cpp,CMakeLists.txt}`;
- `include/leakflow/plot/table_view.hpp`, `src/plot/table_view.cpp`, and
  `tests/plugins/plot/{table_view_test.cpp,CMakeLists.txt}` for generic sorting,
  update, and clear behavior;
- `include/leakflow/plugins/ml_plot`, `src/plugins/ml_plot`, and
  `tests/plugins/ml_plot` trees for `ClusteringMetricsTablePlot` only;
- root `CMakeLists.txt`, `tests/CMakeLists.txt`, `src/apps/CMakeLists.txt`,
  `src/apps/leakflow/leakflow_cli.cpp`, `src/apps/leakflow_ls/main.cpp`, and
  `tests/apps/{CMakeLists.txt,cli_syntax_test.cpp}` for target, descriptor,
  factory, runtime, CLI, and inspect-tool integration; and
- `docs/guides/COOL_PIPELINES.md` for the runnable A4 graph.

Current `clustering_metrics.hpp/.cpp` APIs remain available. Shared kernels may
move only if their public behavior and tests remain compatible.

Deferred follow-up work owns:

- a versioned payload codec and persistence round-trip tests;
- `include/leakflow/plot/metric_view.hpp` and `src/plot/metric_view.cpp`;
- `ClusteringMetricsPlot` and `ClusteringMatrixPlot` additions to the existing
  ML→plot bridge;
- focused generic-view tests in the existing plot test area where appropriate.

No `leakflow_ml` or `leakflow_plugins_ml` file may include plot headers, and no
generic plot file may include an ML payload header. Only the
`leakflow_plugins_ml_plot` bridge sees both contracts.

## 14. Validation Coverage

### Numeric Tests

Cross-validate against scikit-learn for:

- ARI;
- AMI with arithmetic normalization;
- homogeneity;
- completeness;
- V-measure;
- compatibility NMI.

Use hand-computed fixtures for:

- purity;
- pair counts and pair precision/recall/F1;
- micro/macro/per-dimension impurity;
- merge rate and conditional severity;
- micro/macro/per-group fragmentation;
- exact and semantic-cost alignments;
- the optional combined score.

Required scenario coverage:

- perfect partition and arbitrary label permutation;
- pure over-splitting and total merging;
- singleton clusters and singleton truth groups;
- one predicted cluster and one truth group;
- imbalanced groups;
- rectangular \(G<K\) and \(G>K\);
- sparse, negative predicted IDs;
- one, two, and four semantic dimensions;
- batched units and typed-unit reordering/partial overlap;
- \(p=1\) and \(p=2\);
- zero weights, invalid weights/ranges/power, out-of-range values, and NaNs;
- exact-only mode without semantic ranges;
- every undefined denominator;
- a stress case that would expose an accidental \(O(N^2)\) implementation.

A1–A3 numeric tests now cover the conventional metrics, semantic/
fragmentation records, both rectangular alignment directions, exact and
semantic mappings that intentionally differ, a tie that requires an
equality-graph matching flip, both powers, `D=1/2/4`, zero weights, numeric dtype
edges, batching, observation permutation, and large-`N`/small-assignment stress.
The optional combined score is covered with A4's property/payload contract.

### Pipeline Tests

- descriptor/property validation;
- vector-truth shape validation;
- unit alignment;
- optional combined-quality behavior;
- structured payload contents, effective evaluator options, output-`Buffer`
  typed units, and bounded summary;
- deterministic bounded capture of scalar `payload.cluster.*` labels metadata,
  including ignored namespaces and overflow/bounds behavior;
- downstream invalidation after an evaluator property change;
- unchanged legacy `ClusteringStats` behavior.

### A4 Table-Bridge Tests

- deterministic translation from a fixture payload into `TableView` data;
- exact, semantic, fragmentation, alignment, per-dimension/detail, direction,
  support, and undefined-reason rows;
- effective options, typed-unit/count context, captured `payload.cluster.*`
  values, and explicit `payload.parameter.*` input metadata;
- collision-proof `parameter.payload.*` and `parameter.metadata.*` columns;
- `replace|append`, explicit clear, history/reset, and stable per-column
  `asc|desc` sorting for numeric, textual, unavailable, and tied values;
- descriptor/factory/CLI registration in headless operation;
- property-effect and fixture instrumentation proving display changes do not
  call evaluation or assignment code.

ImGui/ImPlot rendering remains manual-only. Manual validation covers one offline
multi-unit table, append/clear/sort controls, and an Idle-state display-property
change.

### Deferred Tests

Codec round trips, `MetricView`/metric-chart data, and matrix/heatmap selection,
normalization, and unmatched annotations belong to their deferred slices, not
A4.

## 15. Exit Criteria

### Implemented A4 / Phase A

- The numeric evaluator returns every required exact and fragmentation result,
  plus every semantic and requested alignment result when enabled, for
  `[N]`/`[N,D]` and
  `[U,N]`/`[U,N,D]`.
- Undefined values, supports, direction, family, and averaging are explicit.
- Reference and hand-computed tests pass.
- The output `Buffer` preserves typed units; the new payload preserves effective
  options, bounded generic producer parameters, structured results, and a
  bounded summary.
- `ClusteringMetricsTablePlot` makes the stored metrics and captured parameters
  inspectable through generic `TableView`, including deterministic
  replace/append, clear, and per-column sorting behavior.
- Table operations over cached data do not rerun evaluation or alignment.
- Existing `ClusteringStats` pipelines and tests remain unchanged and green.
- No plotting dependency enters ML or core targets; the bridge dependency is
  isolated in `leakflow_plugins_ml_plot`, and no clustering-specific branch
  enters `PlotRuntime` or `TableView`.

### Deferred Follow-up

- The payload round-trips through a versioned persistence codec.
- Headline and per-dimension metrics have a generic visual representation.
- Raw and aligned matrices are selectable without recomputation.
- Deferred headless display-data tests pass and manual GUI validation is
  recorded.
- No clustering-specific branch enters `PlotRuntime` or generic plot views.

## Deferred

- Removing `ClusteringStats`.
- Automatically deriving semantic ranges from data.
- Arbitrary real power exponents beyond \(p=1\) and \(p=2\).
- Continuous-label or tolerance-based truth equality.
- GPU-specific kernels.
- Versioned `ClusteringEvaluationPayload` persistence.
- Generic `MetricView`/`ClusteringMetricsPlot` and `ClusteringMatrixPlot`.
- Confidence intervals, bootstrap resampling, persistent temporal metric
  histories, and statistical cross-run comparison beyond A4's in-memory append
  table.
- Using any evaluation metric as a training objective.
