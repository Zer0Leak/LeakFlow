# Distribution-Aware Semantic Fragmentation

Status: **deferred design idea; not implemented and not scheduled**.

This note records a possible extension to LeakFlow's clustering evaluation. It
does not add metric identifiers, result fields, evaluator properties, payload
schema changes, table columns, tests, or roadmap commitments. The implemented
contract remains
[`clustering_evaluation_metrics.md`](clustering_evaluation_metrics.md), and the
implemented metrics remain those listed in
[`CLUSTERING_EVALUATION.md`](../guides/CLUSTERING_EVALUATION.md).

The proposal is clustering-algorithm independent. It must not depend on GMM,
AES, Kyber, Hamming weight, a fixed truth dimension, or a fixed number of
classes. Hamming-weight pairs are useful examples only.

## 1. Motivation

The implemented fragmentation metrics answer whether observations from one
exact truth group were split across predicted clusters. For an unordered pair
of observations with identical truth vectors:

```text
same predicted cluster       -> preserved pair (TP)
different predicted clusters -> fragmented pair (FN)
```

Every fragmented pair currently has the same cost. This is deliberate and
remains useful, but it cannot distinguish these cases:

- fragments land in predicted clusters with nearly identical semantic
  composition;
- fragments land in predicted clusters with very different semantic
  composition.

Semantic impurity does not answer that question. It measures semantic mixing
*inside* predicted clusters, whereas the missing diagnostic concerns the
semantic dispersion *between destinations* receiving fragments of one truth
group.

The desired extension therefore keeps structural fragmentation and adds a
separate answer to:

> When observations from one truth group are split, how different are the
> semantic distributions of the predicted clusters receiving those fragments?

## 2. Why a Centroid Is Insufficient

A first approximation could assign every predicted cluster its mean truth
vector and compare those means. This detects different average destinations,
but equal means do not imply equal distributions.

For example:

```text
predicted cluster A: 50% (2,2), 50% (6,6) -> centroid (4,4)
predicted cluster B: 100% (4,4)            -> centroid (4,4)
```

Centroid distance is zero even though the semantic compositions differ. A
centroid-based score may remain useful as a cheap approximation, but it should
not be the primary definition of distribution-aware fragmentation.

### Centroid approximation

For predicted cluster `k`, define its semantic centroid from truth-vector
composition:

```text
mu[k] = sum_g P_k(g) * h[g]
```

A cheap approximation can replace distribution transport with
`d(mu[k],mu[l])` for two fragment destinations. It supports the same pair
weighting developed below and directly records the original average-HW idea.
It cannot distinguish the equal-centroid example above and therefore must be
named and presented as an approximation, not silently substituted for the
distribution-aware result.

Distance from a truth-group vector `h[g]` to one receiving centroid `mu[k]`
answers a different question: destination displacement or contamination. It
does not measure dispersion *between* fragment destinations and overlaps more
strongly with merge-side semantic impurity. That quantity could be a separately
named diagnostic, but it is not the proposed fragmentation score.

## 3. Existing Quantities

For one evaluated unit:

- `N` is the observation count;
- `G` is the number of observed exact truth groups;
- `K` is the number of observed predicted clusters;
- `n[g,k]` is the contingency count for truth group `g` and predicted cluster
  `k`;
- `n[g] = sum_k n[g,k]`;
- `n[k] = sum_g n[g,k]`;
- `h[g]` is the semantic vector of truth group `g`.

The number of true within-group pairs is:

```text
T = sum_g choose(n[g], 2) = TP + FN
```

The number of fragmented same-truth pairs is:

```text
FN = sum_g sum_(k<l) n[g,k] * n[g,l]
```

The implemented micro fragmentation is:

```text
fragmentation_micro = FN / T
```

All pair counts are unordered and must retain the existing checked 64-bit
semantics.

## 4. Predicted-Cluster Semantic Distributions

Each predicted cluster defines an empirical probability distribution over the
observed exact truth vectors:

```text
P_k(g) = n[g,k] / n[k]
```

The mathematical distribution may be written over all observed truth groups,
including zero-mass entries. Any use of the current Sinkhorn primitive must
first compact `P_k` to its strictly positive support; passing the full vector
with zeros would violate that primitive's input contract.

This distribution comes entirely from the evaluator's existing labels, truth
vectors, and contingency. It does not use feature-space GMM means, posterior
probabilities, or truth during clustering.

The baseline proposal uses the complete inclusive distribution `P_k`, including
the truth group currently being evaluated. Excluding the current truth group
could emphasize surrounding contamination, but it creates empty-distribution
edge cases and changes the question being answered. That alternative remains an
open design decision rather than part of this deferred baseline.

## 5. Semantic Ground Cost

For semantic truth vectors `a` and `b`, reuse the implemented normalized power
cost:

```text
d(a,b)
  = sum_dimension weight[d]
      * (abs(a[d] - b[d]) / range[d])^power
    / sum_dimension weight[d]
```

The cost remains in `[0,1]` under the existing validated semantic contract.
With `power=2`, it is a power cost rather than a mathematical distance because
no outer square root is applied.

Any future implementation must reuse the evaluator's effective ranges,
weights, power, dimension names, validation, and undefined semantics. It must
not infer SCA-specific ranges or class encodings.

## 6. Distribution Transport Cost

For two predicted clusters `k` and `l`, define a normalized semantic transport
cost between `P_k` and `P_l` using `d(h[a], h[b])` as the ground-cost matrix:

```text
transport(k,l)
  = min over couplings pi between P_k and P_l
      sum_(a,b) pi[a,b] * d(h[a], h[b])
```

The intended interpretation is:

- identical semantic distributions have zero transport cost;
- distinct distributions can also have zero cost when they differ only in
  dimensions assigned zero semantic weight, so zero means indistinguishable
  under the configured semantic cost rather than necessarily identical;
- equal centroids with different distributions can have positive cost;
- distributions concentrated on semantically close truth vectors have low
  cost;
- distributions concentrated on distant truth vectors have high cost.

Because the configured power cost need not satisfy metric axioms, the result
should be named a **semantic transport cost**, not unconditionally called a
Wasserstein distance.

### Exact transport versus Sinkhorn

LeakFlow already has a generic log-domain Sinkhorn implementation that returns
a regularized coupling/log-plan. That makes entropic transport a plausible
numeric foundation, but does not by itself provide a Sinkhorn divergence or
settle the metric contract:

- a regularized transport plan's raw ground-cost expectation can be positive
  even when comparing a distribution with itself;
- standard debiasing must use the same complete regularized objective and
  entropy/KL convention for the cross and both self terms; subtracting raw
  ground-cost expectations of independently regularized plans is not the
  standard Sinkhorn divergence and can produce invalid negative artifacts;
- even a correctly defined debiased result adds epsilon semantics, numerical
  tolerances, and possible small residuals, and it must not be assumed to stay
  in `[0,1]` without a proof and explicit bounds contract;
- the current Sinkhorn primitive requires strictly positive marginals, while a
  full `P_k` vector normally contains zeros; each cluster must be compacted to
  its positive truth-group support and compared using a rectangular
  `support(k) x support(l)` ground-cost matrix, or the primitive must be
  deliberately extended;
- an exact discrete transport solver would provide cleaner semantics but is not
  currently part of this proposal.

The choice between exact transport and a carefully specified debiased Sinkhorn
form must be resolved before implementation. The existing `sinkhorn(...)` API
is reusable infrastructure, not authorization to select an unstated epsilon or
silently redefine the score.

## 7. Candidate Metrics

The names below are provisional. All are lower-is-better and require semantic
evaluation.

For compactness, define the transport-weighted fragmentation numerator:

```text
E = sum_g sum_(k<l)
      n[g,k] * n[g,l] * transport(k,l)
```

Only predicted-cluster pairs that jointly receive observations from at least
one truth group can contribute.

### Conditional semantic fragmentation transport severity

```text
conditional_semantic_fragmentation_transport_severity = E / FN
```

This answers:

> Given that a same-truth pair was fragmented, how different were the semantic
> distributions of its two destination clusters?

Its support is `FN`. When `FN == 0`, it is unavailable with a future explicit
reason such as `no_fragmentation_error_pairs`; this absence is normally good.

### Semantic fragmentation transport micro

```text
semantic_fragmentation_transport_micro = E / T
```

Its support is the true-within-group pair count `T`. When `T == 0`, it is
unavailable for the same reason as current pair fragmentation: no truth group
contains a pair.

When `FN > 0`, the two candidate metrics satisfy the diagnostic identity:

```text
semantic_fragmentation_transport_micro
  = fragmentation_micro
    * conditional_semantic_fragmentation_transport_severity
```

This mirrors the implemented merge-side identity:

```text
semantic_impurity_micro
  = merge_error_rate * conditional_merge_error_severity
```

When `FN == 0` and `T > 0`, the micro transport metric is defined as zero but
conditional severity is unavailable, so the identity is not asserted as an
operation over two defined metric values.

### Per-group detail and macro summary

For a non-singleton truth group `g`:

```text
E[g] = sum_(k<l)
         n[g,k] * n[g,l] * transport(k,l)

semantic_fragmentation_transport_per_group[g]
  = E[g] / choose(n[g],2)
```

A macro result could average the defined per-group values equally:

```text
semantic_fragmentation_transport_macro
  = mean over non-singleton truth groups
      semantic_fragmentation_transport_per_group[g]
```

Singleton truth groups remain explicit unavailable detail records. A
non-singleton but unfragmented group has a defined value of zero. Per-group
records should follow the implemented `detail=full` gating; the macro result is
a global summary, not a detail record.

Provisional support and availability rules are:

| Proposed record | Support | Unavailable condition |
|---|---:|---|
| `conditional_semantic_fragmentation_transport_severity` | `FN` | no fragmentation-error pairs |
| `semantic_fragmentation_transport_micro` | `T` | no true within-group pairs |
| `semantic_fragmentation_transport_macro` | eligible non-singleton truth groups | no eligible truth groups |
| `semantic_fragmentation_transport_per_group` | `choose(n[g],2)` | truth group is a singleton |

All four records should carry `semantic_disabled` when semantic evaluation is
off. Any future undefined-reason enum addition, such as
`no_fragmentation_error_pairs`, remains part of the separately approved
implementation contract.

Per-dimension transport contributions are deferred. For exact transport or a
raw cross-plan ground-cost expectation, they require one stored or
deterministically reproduced coupling and clear rules for optimal-coupling
ties; independently optimizing each dimension would describe a different
transport problem. A debiased Sinkhorn definition additionally uses cross and
self objectives, including entropy terms that have no natural per-semantic-
dimension decomposition, so the same attribution rule cannot be assumed.

## 8. Required Joint Interpretation

Distribution-aware semantic fragmentation must not replace structural
fragmentation.

Consider one truth group split between two predicted clusters with identical
semantic distributions:

```text
fragmentation_micro                                      > 0
conditional_semantic_fragmentation_transport_severity   = 0
semantic_fragmentation_transport_micro                   = 0
```

The split is structurally real even though its destinations are semantically
indistinguishable. Conversely, an unfragmented truth group contributes no
fragmentation transport cost even if its one destination cluster is heavily
mixed; merge-side semantic impurity owns that problem.

The intended diagnostic set is therefore:

| Question | Metric family |
|---|---|
| How often are different truth groups merged? | merge error rate / pair precision |
| How semantically severe are those merges? | conditional merge severity / semantic impurity |
| How often are truth groups split? | fragmentation / pair recall |
| How different are the semantic distributions receiving those fragments? | proposed fragmentation transport metrics |

No one row replaces the others.

## 9. Relationship to Semantic Partition Quality

The implemented `semantic_partition_quality` remains unchanged:

```text
semantic separation = semantic_partition_separation
group preservation  = pair_recall
```

This proposal does not add transport severity to that composite. Structural
fragmentation must continue to receive a penalty even when two destination
distributions are identical. Adding a third component or defining another
composite would require separate motivation, repeated-seed evidence, and an
explicitly approved future phase.

The transport metric should first be evaluated as a diagnostic, not as a
training objective or a reason to revise semantic partition quality.

## 10. Computational Direction

A future implementation should operate on the existing sparse contingency and
canonical truth vectors without enumerating observation pairs.

Potential sequence for one unit:

1. Build each `P_k` from cluster marginals and sparse contingency counts,
   compacting it to strictly positive truth-group support for the current
   Sinkhorn primitive.
2. Build the `G x G` semantic ground-cost matrix once.
3. Identify only predicted-cluster pairs that co-occur within at least one
   truth group.
4. Compute or cache one deterministic transport result for each required
   predicted-cluster pair.
5. Aggregate with `n[g,k] * n[g,l]` weights into global and per-group results.

Worst-case transport-pair count is `K*(K-1)/2`, but sparse co-occurrence can be
much smaller. Any implementation plan must bound time and memory explicitly and
include a non-quadratic-in-`N` stress test. Batched units remain independent.

The numeric work belongs in `leakflow_ml`. Pipeline payload and table exposure,
if later approved, belong in the existing ML and ML-plot plugin layers. No
clustering-specific branch belongs in generic plot views.

## 11. Edge Cases and Expected Semantics

- **No true within-group pairs:** micro, macro, and per-group availability
  follow current fragmentation semantics.
- **No fragmented pairs:** conditional severity is unavailable; micro semantic
  transport fragmentation is defined as zero when `T > 0`.
- **Different predicted IDs, identical distributions:** structural
  fragmentation is positive and transport severity is zero.
- **Same centroid, different distributions:** centroid approximation is zero;
  distribution transport can be positive.
- **Pure duplicate fragments:** transport severity is zero; structural
  fragmentation still exposes the split.
- **One predicted cluster:** when `T > 0`, transport fragmentation is zero and
  conditional severity is unavailable; when `T == 0`, both follow the current
  no-true-pair undefined semantics. Merge metrics remain capable of exposing
  contamination.
- **Zero semantic weights:** continue using existing validation that requires a
  positive total weight.
- **Sparse or negative predicted IDs:** canonicalization remains unchanged.
- **Cluster-ID permutation:** every proposed result must remain invariant.
- **Transport ties or numeric residuals:** require deterministic/toleranced
  behavior and must never silently turn a materially invalid value into zero.

## 12. Validation Required Before Any Implementation

A future phase would need hand-computed fixtures covering:

- fragmented pure duplicate clusters: positive structural fragmentation, zero
  transport fragmentation;
- different distributions with the same centroid: positive transport cost;
- nearby versus distant semantic distributions under `power=1` and `power=2`;
- the micro frequency-times-severity identity;
- macro/per-group support and singleton behavior;
- no-fragmentation and no-true-pair undefined cases;
- ranges, weights, `D=1/2/4`, numeric dtypes, arbitrary IDs, batches, and unit
  isolation;
- exact or independently checked transport references;
- deterministic repeated runs and cluster-ID permutation invariance;
- sparse stress coverage proving no observation-pair materialization;
- bounds and tolerances for the selected exact or regularized transport
  definition.

Repeated-seed experiments should establish whether the metric adds useful
information beyond structural fragmentation and semantic impurity. A favorable
result from one clustering configuration is not sufficient justification.

## 13. Open Decisions

The following are intentionally unresolved:

1. Exact discrete transport versus a debiased Sinkhorn-based definition.
2. If Sinkhorn is used, the complete regularized objective convention,
   positive-support compaction, epsilon, convergence, debiasing, bounds, and
   undefined semantics.
3. Inclusive predicted-cluster distributions versus target-group-excluded
   distributions.
4. Final metric names, descriptor family placement, and persisted schema IDs.
5. Whether macro and per-group detail ship in the first implementation slice.
6. Whether per-dimension contributions can share one deterministic coupling.
7. Runtime/memory limits and whether an explicit evaluator property enables the
   additional work.
8. Visualization and comparison-table placement.
9. Whether any later composite is scientifically justified.

Until those decisions are resolved in a separately approved phase, this note
remains an idea record only.

## 14. Explicit Non-Goals of This Note

- No implementation or prototype in shared source.
- No modification of the implemented schema-v6 29-metric inventory.
- No change to `semantic_partition_quality`.
- No use of truth during clustering or GMM fitting.
- No paper-specific threshold or parameter choice.
- No promise that the existing Sinkhorn configuration is appropriate without
  metric-specific validation.
- No roadmap scheduling.
