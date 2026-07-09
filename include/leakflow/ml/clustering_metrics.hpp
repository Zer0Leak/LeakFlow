#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace leakflow::ml {

// Ground-truth clustering evaluation, batched over a leading unit axis. Compares a predicted
// partition (arbitrary cluster ids) against true class labels. The metrics are
// permutation-invariant -- they measure whether same-class points land in the same cluster,
// not whether a cluster is "named" correctly -- so they never match a cluster to a class.
//
// Motivating use: quantify how well a GMM splits SCA traces into the true (hm, hy) classes
// (class = (B+1)*hm + hy) without the GMM ever being told the class. The primitive is
// domain-agnostic.

struct ClusteringMetrics {
    torch::Tensor confusion;              // [U, C, K] counts: true class c x predicted cluster k
    torch::Tensor purity;                 // [U] weighted mean cluster purity, in [0, 1]
    torch::Tensor adjusted_rand_index;    // [U] chance-adjusted agreement, <=1 (1 = identical)
    torch::Tensor normalized_mutual_info; // [U] in [0, 1] (arithmetic normalisation, sklearn default)
    bool batched = true;                  // false when the caller passed [T]
};

// pred, truth: [T] or [U, T] int64 cluster ids / class labels in [0, n_clusters/n_classes).
[[nodiscard]] torch::Tensor confusion_matrix(
    const torch::Tensor& pred,
    const torch::Tensor& truth,
    std::int64_t n_classes,
    std::int64_t n_clusters); // -> [U, C, K] (or [C, K])

// The three scores below take a confusion matrix [C, K] or [U, C, K] and return [] or [U].
[[nodiscard]] torch::Tensor cluster_purity(const torch::Tensor& confusion);
[[nodiscard]] torch::Tensor adjusted_rand_index(const torch::Tensor& confusion);
[[nodiscard]] torch::Tensor normalized_mutual_info(const torch::Tensor& confusion);

// Convenience: everything at once from label vectors.
[[nodiscard]] ClusteringMetrics clustering_metrics(
    const torch::Tensor& pred,
    const torch::Tensor& truth,
    std::int64_t n_classes,
    std::int64_t n_clusters);

// --- Matching-based metrics (need a cluster<->class correspondence) ---
//
// Cluster ids are arbitrary, so before a diagonal-readable confusion plot or an accuracy
// number you must pair each class with a distinct cluster. hungarian_match solves the optimal
// 1-to-1 assignment (Kuhn-Munkres) that maximises total overlap. Requires a square confusion
// (n_classes == n_clusters).

// confusion [C, K] or [U, C, K] -> matching [C] or [U, C] int64: the cluster paired with each
// class. Requires C == K.
[[nodiscard]] torch::Tensor hungarian_match(const torch::Tensor& confusion);

// Reorder the cluster (last) axis so class c's matched cluster becomes column c, making the
// diagonal the matched overlap. confusion [.,C,K] + matching [.,C] -> [.,C,C].
[[nodiscard]] torch::Tensor reorder_confusion_columns(
    const torch::Tensor& confusion,
    const torch::Tensor& matching);

struct MatchedClusteringScores {
    torch::Tensor matching;  // [U, C] cluster paired with each class
    torch::Tensor accuracy;  // [U] matched-diagonal / N (needs the matching)
    torch::Tensor precision; // [U, C] per matched pair: |c n k| / |k|
    torch::Tensor recall;    // [U, C] per matched pair: |c n k| / |c|
    torch::Tensor f1;        // [U, C] harmonic mean of precision and recall
    bool batched = true;
};

// Hungarian matching + accuracy + per-class precision/recall/F1 from a confusion matrix.
[[nodiscard]] MatchedClusteringScores matched_clustering_scores(const torch::Tensor& confusion);

} // namespace leakflow::ml
