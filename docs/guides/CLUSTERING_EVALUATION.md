# Interpreting Clustering Evaluation Metrics

This guide explains how to read every metric produced by
`ClusteringEvaluate` and displayed by `ClusteringMetricsTablePlot`. It is the
practical companion to the mathematical design record in
[`docs/design/clustering_evaluation_metrics.md`](../design/clustering_evaluation_metrics.md).
For a complete runnable graph, see
[`docs/guides/COOL_PIPELINES.md`](COOL_PIPELINES.md#compare-poi-transfer-and-gmm-clusters).

The evaluator is clustering-algorithm independent. The examples use GMM labels
and the semantic truth vector `(HW(m), HW(y))`, but the same interpretation
applies to other predicted cluster IDs and vector-valued truth.

## The Short Version

Start with a small set of complementary metrics:

| Question | Start with | Direction |
|---|---|---:|
| Does the complete predicted partition agree with truth? | `adjusted_rand_index` | ↑ |
| Are different truth groups being merged? | `pair_precision`, `merge_error_rate` | ↑, ↓ |
| How severe are those merges in semantic coordinates? | `semantic_impurity_micro`, `conditional_merge_error_severity` | ↓ |
| Are truth groups being split across clusters? | `pair_recall`, `fragmentation_micro` | ↑, ↓ |
| Is one particular group or cluster causing the problem? | per-group fragmentation and per-cluster semantic impurity | ↓ |
| How good is the best one-to-one exact mapping? | `exact_alignment_matched_accuracy` | ↑ |
| Is one compact scalar needed for sorting otherwise comparable runs? | `combined_quality` | ↑ |

No single metric is sufficient. In particular:

- purity can look excellent when the clustering over-splits truth groups;
- homogeneity can look excellent while completeness is poor;
- an exact-aligned Heatmap shows the selected mapping, not the complete quality
  of the partition; when `G > K`, unmatched truth rows also prevent every
  selected pair from occupying the geometric diagonal;
- combined quality is useful for ranking comparable runs, not as a replacement
  for its semantic-impurity and fragmentation components.

## Vocabulary

For one evaluated unit:

- `N` is the number of observations.
- `S`, when displayed, is the number of input features fitted by the clustering
  producer. It is `N/A` when the producer does not report that context.
- A **truth vector** is the semantic value attached to an observation, such as
  `(hm=3, hy=5)`.
- A **truth group** contains observations whose complete truth vectors are
  exactly equal.
- `G` is the number of distinct truth groups actually observed.
- A **predicted cluster** is one label emitted by the clustering algorithm.
- `K` is the number of distinct predicted cluster IDs actually observed.
- `n[g,k]` is the number of observations from truth group `g` placed in
  predicted cluster `k`.

`G` counts observed groups, not every theoretically possible group. For two
byte Hamming weights there are up to `9 × 9 = 81` vectors, but a subset of the
dataset may contain only 45 of them. Similarly, `GaussianMixture(n_components=49)`
requests 49 components, while `K` reports the distinct predicted labels that
actually occur in the evaluated output.

The contingency matrix is:

```text
                         predicted cluster
                     k=0   k=1   ...   k=K-1
true group  g=0     n[0,0] n[0,1] ...
            g=1     n[1,0] n[1,1] ...
            ...
            g=G-1
```

## Direction, Scope, Support, and `N/A`

Metric names in the table include their direction:

- `↑`: higher is better.
- `↓`: lower is better.

The sort arrow on a table header is only the current sort direction; it is not
the metric direction.

### Averaging scopes

- **Micro** combines underlying observations or observation pairs before
  calculating the value. Large groups or clusters can contribute more.
- **Macro** calculates one value per eligible group or cluster and averages
  those values equally.
- **Per cluster** identifies one predicted cluster.
- **Per group** identifies one exact truth group.
- **Per dimension** identifies one semantic coordinate, such as `hm` or `hy`.

Micro and macro answer different questions. Neither is automatically superior:

- for pair-supported micro metrics, use micro to describe the error experienced
  by a randomly selected supported observation pair;
- use macro to describe the typical eligible group or cluster without letting
  the largest ones dominate.

### Support

Every metric carries a support count. Its meaning follows the metric:

- observation-supported values use observations;
- pair metrics use unordered observation pairs;
- macro values use the number of eligible groups or clusters;
- per-group or per-cluster values use that record's applicable observations or
  pairs.

Do not compare two values without noticing a large support difference.

### Undefined values

`N/A` means the value is unavailable, often because its denominator does not
exist. It is not silently converted to zero.

| Reason | Typical situation |
|---|---|
| `no_predicted_within_cluster_pairs` | Every predicted cluster is a singleton, so cluster-pair metrics have no denominator. |
| `no_true_within_group_pairs` | Every truth group is a singleton, so truth-pair recall/fragmentation has no denominator. |
| `no_eligible_predicted_clusters` | No predicted cluster contains at least two observations. |
| `no_merge_error_pairs` | There are no wrongly merged pairs; conditional severity has nothing to average. This is normally good. |
| `no_eligible_truth_groups` | No truth group contains at least two observations. |
| `no_matched_predicted_cluster` | A rectangular exact alignment left this truth group unmatched. |
| `semantic_disabled` | Evaluation used `semantic=off`. |
| `dependent_metric_undefined` | A derived metric cannot be calculated because one of its inputs is undefined. |

## Pair Language: TP, FP, and FN

Many metrics become easier to understand by considering unordered pairs of
observations:

- `TP`: same predicted cluster and same truth group.
- `FP`: same predicted cluster but different truth groups — a **merge error**.
- `FN`: different predicted clusters but the same truth group — a
  **fragmentation error**.

This vocabulary does not require a Hungarian assignment.

## Exact Partition Metrics

Exact metrics treat every distinct truth vector as a different categorical
group. They do not consider whether two different HW vectors are close or far
apart.

### `adjusted_rand_index` (ARI) ↑

ARI compares all observation pairs and adjusts the agreement expected by
chance.

- `1`: identical partitions, ignoring cluster-ID permutation.
- approximately `0`: random-like agreement under the adjustment model.
- negative: worse than the chance-adjusted reference.

ARI is the recommended primary conventional whole-partition score. It penalizes
both merging and splitting, but use the directional metrics below to diagnose
which problem occurred.

### `adjusted_mutual_information` (AMI) ↑

AMI measures shared information between the predicted and truth partitions and
adjusts for chance.

- `1`: perfect agreement.
- approximately `0`: chance-level shared information.
- it may be negative.

AMI is another robust whole-partition score. ARI and AMI emphasize different
structures, so small disagreements between them are normal.

### `homogeneity` ↑

Homogeneity asks:

> Does each predicted cluster contain observations from only one truth group?

- `1`: every predicted cluster is exact-group pure.
- low values: predicted clusters merge truth groups.

Homogeneity does not sufficiently penalize over-splitting. Giving every
observation its own cluster is homogeneous.

### `completeness` ↑

Completeness asks:

> Are all observations from one truth group kept in the same predicted cluster?

- `1`: each truth group stays together.
- low values: truth groups are fragmented across clusters.

Putting everything into one cluster is complete, so completeness must be read
with homogeneity.

### `v_measure` ↑

V-measure is the harmonic mean of homogeneity and completeness with the current
`beta=1` convention. It becomes small when either merging or splitting is bad.

### `normalized_mutual_information` (NMI) ↑

NMI is retained for compatibility. With LeakFlow's arithmetic normalization it
is equivalent to `v_measure` with `beta=1`; do not treat the two values as
independent evidence.

Unlike AMI, NMI does not adjust for agreement expected by chance.

### `purity` ↑

For every predicted cluster, purity keeps the largest truth-group count and
divides the sum of those dominant counts by `N`.

- `1`: every predicted cluster contains only one truth group.
- lower: clusters contain observations outside their dominant truth group.

Purity ignores semantic distance and is optimistic under over-clustering. A
truth group split into many individually pure clusters can produce purity `1`
while fragmentation is severe.

### `pair_precision` ↑

```text
pair_precision = TP / (TP + FP)
```

Of all observation pairs placed together by the clustering, how many truly
belong together?

- high: little exact mixing;
- low: many different truth groups are merged.

It is related to merge detection but is not the same as purity.

### `pair_recall` ↑

```text
pair_recall = TP / (TP + FN)
```

Of all observation pairs that truly belong together, how many did the
clustering keep together?

- high: little fragmentation;
- low: truth groups are split across clusters.

LeakFlow's micro fragmentation is exactly:

```text
fragmentation_micro = 1 - pair_recall
```

### `pair_f1` ↑

Pair F1 is the harmonic mean of pair precision and pair recall. It balances
exact merge and split errors without requiring a label assignment.

## Semantic Power Cost

Exact metrics treat `(hm=3, hy=3)` versus `(hm=3, hy=4)` as just as wrong as
`(hm=0, hy=0)` versus `(hm=8, hy=8)`. Semantic metrics distinguish them.

For truth vectors `a` and `b`, ranges `R[d]`, weights `w[d]`, and
`power = p`:

```text
semantic_cost(a,b)
  = sum_d w[d] * (abs(a[d] - b[d]) / R[d])^p
    / sum_d w[d]
```

The result is in `[0,1]`. It is called a power **cost**, not necessarily a
mathematical distance when `p=2` because the outer square root is deliberately
omitted.

For the example configuration:

```text
semantic_ranges=[8,8]
semantic_weights omitted -> equal weights
power omitted             -> power=2
```

the cost is:

```text
((delta hm)^2 + (delta hy)^2) / 128
```

Examples:

| Truth-pair comparison | Cost |
|---|---:|
| `(3,3)` versus `(3,4)` | `1/128 ≈ 0.0078` |
| `(2,2)` versus `(4,2)` | `4/128 = 0.03125` |
| `(0,0)` versus `(8,8)` | `1.0` |

Semantic scores are comparable only when ranges, weights, power, truth
dimensions, and units have the same meaning.

## Semantic Merge Metrics

All semantic impurity metrics are lower-is-better.

### `merge_error_rate` ↓

```text
merge_error_rate = FP / (TP + FP) = 1 - pair_precision
```

This measures how frequently two observations placed in the same predicted
cluster belong to different exact truth groups. It counts every different
truth pair as wrong but does not measure how far apart the truth vectors are.

This is pair-weighted, not the fraction of predicted clusters that are mixed.

### `conditional_merge_error_severity` ↓

This averages semantic cost only across the `FP` pairs that are actually merge
errors.

- low: the wrong merges are semantically close;
- high: the wrong merges join distant semantic vectors;
- `N/A` with `no_merge_error_pairs`: there are no merge errors to score.

It measures severity, not frequency.

### `semantic_impurity_micro` ↓

Micro semantic impurity averages semantic cost across every within-predicted-
cluster observation pair. Large clusters contribute more because they contain
more pairs.

When merge errors exist:

```text
semantic_impurity_micro
  = merge_error_rate * conditional_merge_error_severity
```

This is the best single answer to:

> Considering both how often incompatible observations are merged and how far
> apart their semantic values are, how bad is the clustering?

It is also the semantic component used by `combined_quality`.

### `semantic_impurity_macro` ↓

Macro semantic impurity calculates one semantic impurity value for each
non-singleton predicted cluster and averages eligible clusters equally.

Use it when a small problematic cluster should matter as much as a large one.
Compare it with micro impurity:

- `macro >> micro`: smaller clusters tend to be worse;
- `micro >> macro`: larger clusters tend to be worse;
- similar values: error severity is more evenly distributed.

Macro impurity is not the fraction of mixed clusters.

### `semantic_impurity_per_cluster` ↓

For one non-singleton predicted cluster, this is the mean semantic cost across
all observation pairs inside that cluster.

- `0`: all supported within-cluster pairs have zero configured semantic cost;
- positive: at least some within-cluster pairs differ in a positively weighted
  semantic dimension;
- `N/A`: the cluster is a singleton and has no pair denominator.

With `detail=full`, the Semantic tab contains one record per predicted cluster.
This is the best current view for finding the specific clusters responsible for
semantic mixing.

LeakFlow does not currently expose a separate headline
`mixed_cluster_count / predicted_cluster_count`. Counting positive per-cluster
impurity records can approximate that question when all relevant dimensions
have positive weights, but it is not a stored metric.

### `semantic_impurity_dimension_micro` ↓

This is micro impurity for one semantic coordinate, such as `hm` or `hy`.
It shows which coordinate contributes raw normalized error. The displayed
per-dimension value is not multiplied by the configured dimension weight.

### `semantic_impurity_dimension_macro` ↓

This is the corresponding per-dimension value averaged equally across eligible
predicted clusters.

Use micro/macro dimension pairs to distinguish, for example, whether `hm`
mixing is concentrated in large or small clusters.

## Fragmentation Metrics

Fragmentation asks whether observations with the same complete truth vector are
split across predicted clusters. All fragmentation metrics are lower-is-better.

### `fragmentation_micro` ↓

```text
fragmentation_micro = FN / (TP + FN) = 1 - pair_recall
```

This is the probability that a supported same-truth observation pair was split
across predicted clusters.

It is the group-preservation component used by `combined_quality`.

### `fragmentation_macro` ↓

For every non-singleton truth group, calculate its fragmentation, then average
eligible truth groups equally.

Compare macro with micro to detect whether small or large truth groups are more
fragmented.

### `fragmentation_per_group` ↓

This reports fragmentation for one exact truth group.

- `0`: every observation from that truth group stayed in one cluster;
- close to `1`: same-group observation pairs were almost always separated;
- `N/A`: the truth group is a singleton.

With `detail=full`, sort these records descending to find the worst preserved
truth groups.

A distribution-aware semantic fragmentation metric has been discussed but is
not implemented. Its deferred design note explains why predicted-cluster
centroids are insufficient and considers semantic transport between complete
truth-vector distributions:
[`docs/design/clustering_semantic_fragmentation.md`](../design/clustering_semantic_fragmentation.md).

## Combined Quality

`combined_quality` is optional and requires semantic evaluation. Define:

```text
semantic cohesion  C = 1 - semantic_impurity_micro
group preservation G = 1 - fragmentation_micro = pair_recall

combined_quality Q = 2*C*G / (C+G)
```

`Q` is in `[0,1]` and higher is better. It is a harmonic mean, so one weak
component strongly reduces it.

- `1`: perfect semantic cohesion and group preservation;
- `0`: at least one component is at its worst endpoint;
- values near zero indicate weak combined structure.

There is no universal threshold for “good.” Compare against baselines and other
runs with the same data and semantic configuration. A value of `0.10` is better
than `0.07`, but both are low in absolute terms.

Always inspect the copied component rows beside the score. For example, a low
score can mean:

- clusters are semantically coherent but truth groups are heavily fragmented;
- truth groups stay together but clusters merge semantically distant groups;
- both problems are present.

Combined quality does not use Hungarian alignment and should not be interpreted
as matched accuracy.

## Alignment Metrics

Core partition, semantic-impurity, and fragmentation metrics are invariant to
cluster-ID permutation. They do not need an assignment. Alignment is optional
post-processing for interpretation and visualization.

LeakFlow can calculate two separate rectangular Hungarian assignments:

- **Exact-overlap alignment** maximizes total raw contingency overlap.
- **Semantic-cost alignment** minimizes configured semantic cost, including a
  fixed maximum dummy penalty for unmatched predicted clusters. Unmatched truth
  groups have no observation mass and add no semantic cost.

With `alignment=both`, both mappings are stored. They can differ. The current
same-window Heatmap uses the stored **exact-overlap** column order; it does not
display the semantic-cost alignment.

### `exact_alignment_matched_accuracy` ↑

The exact Hungarian assignment chooses at most one predicted cluster for each
truth group and at most one truth group for each predicted cluster. Matched
accuracy is:

```text
sum of raw counts in the selected exact-overlap cells / N
```

It measures the best global one-to-one exact mapping. It is not purity: purity
allows several predicted clusters to choose the same dominant truth group,
while Hungarian alignment does not.

### `exact_alignment_precision_per_group` ↑

For a truth group and its assigned predicted cluster:

```text
overlap / predicted-cluster observations
```

This asks how much of the assigned cluster belongs to that truth group.

### `exact_alignment_recall_per_group` ↑

```text
overlap / truth-group observations
```

This asks how much of the truth group was captured by its assigned cluster.

### `exact_alignment_f1_per_group` ↑

The harmonic mean of the assigned pair's precision and recall.

### `exact_alignment_jaccard_per_group` ↑

```text
overlap / (truth support + predicted support - overlap)
```

Jaccard measures intersection over union for the assigned truth-group/cluster
pair.

### `semantic_alignment_cost` ↓

This is the global normalized cost of the separate semantic Hungarian mapping.
It answers how semantically expensive the best one-to-one mapping is, including
maximum penalties for observations in unmatched predicted clusters.

Do not use the exact Heatmap diagonal as if it represented this mapping; the two
assignment objectives can choose different pairs.

### `semantic_alignment_dimension_error` ↓

This splits semantic alignment error by semantic dimension. The displayed
dimension errors are raw and unweighted. To determine contribution to the
aggregate assignment cost, compare `w[d] * dimension_error[d]` using the same
relative weights as the aggregate cost. A zero-weight dimension can therefore
show error without affecting `semantic_alignment_cost`.

## Rectangular Alignment and Unmatched Entries

For `G` truth groups and `K` predicted clusters, assignment uses a padded square
problem of size `max(G,K)`.

- `K > G`: every observed truth group receives a one-to-one predicted partner;
  `K-G` predicted clusters are unmatched.
- `G > K`: every predicted cluster receives a truth-group partner;
  `G-K` truth groups are unmatched.

An unmatched predicted cluster can contain many observations. “Unmatched” means
there was no separate one-to-one truth slot available; it does not mean unused
or empty.

A formal real-to-real partner can also have zero contingency overlap. Hungarian
alignment always completes the rectangular one-to-one assignment; a selected
pair is not evidence of a successful match unless its overlap and aligned
scores are positive.

### Worked shape: 45 truth groups and 49 clusters

The exact-overlap mapping selects 45 one-to-one pairs and leaves four predicted
clusters unmatched:

```text
45 observed truth groups -> 45 matched predicted clusters
                             4 unmatched predicted clusters
```

In the exact-aligned Heatmap:

- displayed columns `0..44` are the clusters selected for truth-group rows
  `0..44`;
- columns `45..48` are unmatched predicted clusters;
- the column label is the actual predicted cluster ID, not merely its displayed
  position.

The first 45 diagonal cells therefore show the selected exact Hungarian pairs.
The assignment maximizes their **raw count sum globally**. An individual
diagonal cell need not be the largest cell in its row because one predicted
cluster cannot be assigned to two truth groups.

The four unmatched clusters often reveal over-splitting. Inspect their column
composition and the affected truth groups' fragmentation records.

## Reading the Heatmap

The Heatmap tab uses:

- rows: canonical truth groups, labeled by complete semantic vectors;
- columns: actual predicted cluster IDs;
- color: the contingency count normalized independently within each truth-group
  row.

Each row's colors sum to 100%. The color answers:

> Where did observations from this truth group go?

It does not show the absolute size of the truth group.

Hovering one cell reports:

| Hover value | Formula | Meaning |
|---|---:|---|
| `observations` | `n[g,k]` | Raw observations in this truth-group/cluster cell. |
| `within true group` | `n[g,k] / sum_k n[g,k]` | Fraction of the truth group captured by this cluster. |
| `within predicted cluster` | `n[g,k] / sum_g n[g,k]` | Fraction of the cluster composed of this truth group. |
| `of all observations` | `n[g,k] / N` | Fraction of the evaluated dataset in this cell. |

When the caption says `exact-overlap aligned contingency` and `K >= G`, the main
matched diagonal represents the exact Hungarian-selected pairs. When `G > K`,
unmatched truth rows have no materialized dummy columns, so some selected cells
can be off the geometric diagonal. In either shape, inspect the selected cells
and both row and column percentages:

- high row and high column percentages: strong one-to-one match;
- high row but low column percentage: the cluster captures the group but is
  contaminated by others;
- low row but high column percentage: a pure fragment of a truth group;
- multiple bright cells in one row: fragmentation;
- one bright column across several rows: merging.

When the caption says `raw contingency`, predicted cluster IDs are arbitrary and
diagonal position has no special meaning.

The Hungarian objective uses raw counts, not row-normalized colors.

## Reading the Eight Tabs

### Overview

Start here when comparing runs. It contains one row per run and unit with:

- `Observations (N)` and `Features (S)`;
- observed truth-group and predicted-cluster counts;
- selected headline exact, semantic, fragmentation, and combined metrics;
- captured clustering and explicitly stamped experiment parameters.

Never compare rows without checking that the data, unit, N, semantic options,
and relevant producer parameters are comparable.

### Exact

Contains all conventional partition metrics. Use ARI/AMI for overall agreement,
then homogeneity/completeness or pair precision/recall to diagnose merges versus
splits.

### Semantic

Contains global, per-dimension, and—when `detail=full`—per-cluster semantic
records. Use merge rate for frequency, conditional severity for semantic
distance, and micro impurity for their combined effect.

### Fragmentation

Contains micro, macro, and—when `detail=full`—per-truth-group fragmentation.

### Combined

Contains combined quality and its copied semantic-impurity and fragmentation
components. It is intended for sorting otherwise comparable experiments.

### Alignment

Contains exact matched accuracy, exact per-group scores, semantic alignment
cost, and semantic per-dimension alignment errors when requested.

### Heatmap

Contains the stored full-detail contingency, exact-aligned when an exact mapping
is available and raw otherwise. It never recomputes an assignment.

### Parameters

Contains effective evaluator options, bounded clustering-producer context, and
explicit `payload.parameter.*` experiment metadata. Use it to verify that two
accumulated rows are genuinely comparable.

## Common Clustering Failure Patterns

### Too few predicted clusters: under-clustering

Typical signs:

- homogeneity and pair precision decrease;
- merge-error rate and semantic impurity increase;
- completeness and pair recall may remain deceptively high;
- Heatmap columns receive mass from several truth-group rows.

### Too many predicted clusters: over-clustering

Typical signs:

- purity and homogeneity may remain high;
- completeness and pair recall decrease;
- fragmentation increases;
- exact alignment leaves extra predicted clusters unmatched when `K > G`;
- one truth-group row spreads across several columns.

### Frequent but nearby semantic merges

Typical signs:

- merge-error rate is high;
- conditional merge-error severity is low;
- semantic micro impurity is moderated by the small distances.

### Rare but severe semantic merges

Typical signs:

- merge-error rate is low;
- conditional severity is high;
- per-cluster and per-dimension detail identify the responsible clusters and
  coordinates.

### Good-looking aligned diagonal but weak clustering

The Hungarian mapping always chooses the best global one-to-one pairing. A
visible diagonal can coexist with:

- substantial off-diagonal mass;
- low exact matched accuracy;
- high fragmentation in extra/unmatched clusters;
- high cluster contamination.

Use the diagonal as a mapping aid, not a standalone quality metric.

## Comparing GMM Configurations

For accumulated runs, use this workflow:

1. Confirm the same dataset, unit, N, truth definition, and feature selection.
2. Confirm identical semantic ranges, weights, and power.
3. Compare ARI and AMI for whole-partition agreement.
4. Compare pair precision/merge-error rate for merges.
5. Compare pair recall/fragmentation for splits.
6. Compare semantic impurity and conditional severity for HW-aware merge cost.
7. Use combined quality only after inspecting its two components.
8. Inspect exact matched accuracy and the worst per-group alignment rows.
9. Use the Heatmap and per-cluster/per-group detail to localize the problem.

When changing `n_components`, expect tradeoffs:

- increasing components can improve purity while worsening fragmentation;
- decreasing components can improve preservation while worsening semantic
  mixing.

When changing semantic `power`, ranges, or weights, the semantic metrics and
combined quality change meaning. Do not rank those rows as though the scale were
unchanged.

## Practical Baselines and “Is This Good?”

LeakFlow intentionally does not define universal thresholds. A useful baseline
set is:

- random labels with the same cluster-size distribution;
- a trivial one-cluster result;
- a deliberately over-split result;
- the best result from a known profiling configuration;
- repeated GMM seeds or initializations.

Prefer differences that are stable across repeated runs. A small improvement in
one metric may simply exchange merging for fragmentation; the directional
metric pair reveals that tradeoff.

## Evaluator Options That Change Available Metrics

The example uses:

```text
ClusteringEvaluate(
  semantic=power,
  semantic_ranges=[8,8],
  dimension_names=[hm,hy],
  detail=full,
  alignment=both,
  combined_quality=true
)
```

- `semantic=off|power`: enables or disables semantic-cost metrics.
- `semantic_ranges=[...]`: required normalization range for each semantic
  dimension in power mode.
- `semantic_weights=[...]`: optional nonnegative dimension weights; equal by
  default.
- `power=1|2`: linear or squared normalized power cost; default `2`.
- `dimension_names=[...]`: labels dimensions in tables and truth vectors.
- `detail=global|full`: Full adds per-cluster, per-group, contingency, and other
  detailed records needed by the Heatmap.
- `alignment=none|exact|semantic|both`: selects optional stored assignments.
- `combined_quality=true|false`: requests the optional harmonic composite.

`alignment=semantic|both` and `combined_quality=true` require
`semantic=power`; invalid combinations are rejected rather than producing
misleading values.

## Complete Metric Inventory

The implemented descriptor inventory contains 28 metrics:

### Exact

1. `adjusted_rand_index`
2. `adjusted_mutual_information`
3. `homogeneity`
4. `completeness`
5. `v_measure`
6. `purity`
7. `pair_precision`
8. `pair_recall`
9. `pair_f1`
10. `normalized_mutual_information`

### Semantic

11. `semantic_impurity_micro`
12. `semantic_impurity_macro`
13. `merge_error_rate`
14. `conditional_merge_error_severity`
15. `semantic_impurity_dimension_micro`
16. `semantic_impurity_dimension_macro`
17. `semantic_impurity_per_cluster`

### Fragmentation

18. `fragmentation_micro`
19. `fragmentation_macro`
20. `fragmentation_per_group`

### Alignment

21. `exact_alignment_matched_accuracy`
22. `exact_alignment_precision_per_group`
23. `exact_alignment_recall_per_group`
24. `exact_alignment_f1_per_group`
25. `exact_alignment_jaccard_per_group`
26. `semantic_alignment_cost`
27. `semantic_alignment_dimension_error`

### Combined

28. `combined_quality`

The design record remains authoritative for exact formulas, result structures,
tie-breaking, validation, and implementation contracts. This guide is the
interpretation layer for experiments.
