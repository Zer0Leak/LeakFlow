# GMM comparison: LeakFlow 49 vs original-code 81

## Scope

This is a controlled, single-seed comparison run on July 17, 2026. It is **not**
a result reported in the paper and it does not establish that either
implementation is generally better. It runs the GMM construction used by the
ChipWhisperer branch of the authors' original Python code and compares its
labels with LeakFlow labels on exactly the same feature matrix.

The original implementation constructs 100 features by concatenating 50 CPA
PoIs for `HW(m)` and 50 CPA PoIs for `HW(y)`, then calls scikit-learn's
`GaussianMixture(n_components=81)`. The source call is in
`Classic_blind_attack_kyber_chipwhisperer.py` in the supplied original-code
archive.

## Controlled setup

Both fits used:

- the original synchronized ChipWhisperer arrays (`traces.npy`, `plain.npy`,
  `labels.npy`, and `key.npy`);
- AES byte 0 with fixed key byte `0x2b`;
- the first 10,000 traces, with Gaussian noise standard deviation `0.05` and
  NumPy seed 0;
- the first 8,000 observations for fitting and evaluation;
- the same top 50 absolute-CPA `HW(m)` PoIs and top 50 absolute-CPA `HW(y)`
  PoIs;
- one `[8000, 100]` `float64` feature matrix and the same 45 observed
  `(HW(m), HW(y))` truth groups; and
- semantic ranges `[8, 8]`, equal weights, and power 2 for LeakFlow's semantic
  metrics.

The fits intentionally differ as requested:

| Configuration | Components | Covariance | Initialization | Restarts | Max iterations | Seed |
|---|---:|---|---|---:|---:|---:|
| LeakFlow | 49 | diagonal | `kmeans++` | 1 | 100 | 0 |
| Original-code scikit-learn | 81 | full | `kmeans` | 1 | 100 | 0 |

The authors' source omits `random_state`; this run adds `random_state=0` so the
comparison can be repeated. The remaining scikit-learn defaults used here were
`tol=1e-3` and `reg_covar=1e-6`. scikit-learn 1.8 was used.

## Results

Higher is better except for semantic impurity.

| Metric | LeakFlow, K=49 | Original code, K=81 |
|---|---:|---:|
| Adjusted Rand index | 0.000755 | **0.000846** |
| Adjusted mutual information | 0.008173 | **0.010842** |
| Homogeneity | 0.050371 | **0.079348** |
| Completeness | 0.044354 | **0.062083** |
| V-measure / NMI | 0.047172 | **0.069662** |
| Purity | 0.091750 | **0.102750** |
| Pair precision | 0.042490 | **0.043141** |
| Pair recall | **0.021566** | 0.013488 |
| Pair F1 | **0.028610** | 0.020551 |
| Semantic impurity | 0.060560 | **0.059313** |
| Semantic partition separation | 0.979636 | **0.987714** |
| Semantic partition quality | **0.042202** | 0.026613 |

The underlying pair counts make the fragmentation tradeoff explicit:

| Count | LeakFlow, K=49 | Original code, K=81 |
|---|---:|---:|
| True-positive pairs | 28,567 | 17,867 |
| False-positive pairs | 643,750 | 396,283 |
| False-negative pairs | 1,296,089 | 1,306,789 |
| True within-group pairs | 1,324,656 | 1,324,656 |

Relative to the 81-component result, LeakFlow's 49-component result has 58.58%
higher semantic partition quality, 59.89% higher pair recall, and 39.22% higher
pair F1. The 81-component result has better ARI, AMI, V-measure, purity,
semantic impurity, and semantic partition separation. Purity is especially
unsafe as a headline comparison here because increasing the number of clusters
can raise it through fragmentation.

Both clusterings are weak in absolute terms: ARI is below 0.001 and pair F1 is
below 0.03 in both runs. The useful conclusion is therefore narrow: on this
seed and feature matrix, 49 components preserve more same-truth pairs and obtain
the better balanced semantic partition score, while 81 components separate a
slightly larger fraction of semantic variation and score better on the listed
information-theoretic metrics.

`semantic_partition_quality` is the cross-`K` score used for this comparison.

## Timing note

Observed process/fit times on the same machine were approximately 30.93 seconds
for the LeakFlow diagonal fit and 2.10--3.04 seconds for the scikit-learn fit. A
LeakFlow full-covariance, 49-component control took approximately 1.11 seconds
and produced semantic partition quality 0.044112.

These timings are diagnostic only, not an engine benchmark: component count,
covariance model, initialization, and the timed code boundaries differ.
