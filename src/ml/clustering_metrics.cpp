#include "leakflow/ml/clustering_metrics.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

namespace leakflow::ml {
namespace {

struct PreparedLabels {
    torch::Tensor labels; // [U, T] int64
    bool batched;
};

[[nodiscard]] PreparedLabels prepare_labels(const torch::Tensor& labels, const char* what)
{
    auto working = labels.to(torch::kLong).contiguous();
    if (working.dim() == 1) {
        return {working.unsqueeze(0), false};
    }
    if (working.dim() == 2) {
        return {working, true};
    }
    throw std::invalid_argument(std::string("clustering_metrics ") + what + " must be [T] or [U, T]");
}

struct PreparedConfusion {
    torch::Tensor confusion; // [U, C, K] float64
    bool batched;
};

[[nodiscard]] PreparedConfusion prepare_confusion(const torch::Tensor& confusion)
{
    auto working = confusion.to(torch::kFloat64);
    if (working.dim() == 2) {
        return {working.unsqueeze(0), false};
    }
    if (working.dim() == 3) {
        return {working, true};
    }
    throw std::invalid_argument("clustering_metrics confusion must be [C, K] or [U, C, K]");
}

// n choose 2 for real-valued counts.
[[nodiscard]] torch::Tensor comb2(const torch::Tensor& counts)
{
    return counts * (counts - 1.0) * 0.5;
}

[[nodiscard]] torch::Tensor drop_unit_axis(const torch::Tensor& tensor, bool batched)
{
    return batched ? tensor : tensor.squeeze(0);
}

// -sum p log p over the last axis, treating p == 0 as contributing 0.
[[nodiscard]] torch::Tensor entropy(const torch::Tensor& probabilities)
{
    const auto terms = torch::where(
        probabilities > 0, probabilities * torch::log(probabilities), torch::zeros_like(probabilities));
    return -terms.sum(-1);
}

} // namespace

torch::Tensor confusion_matrix(
    const torch::Tensor& pred,
    const torch::Tensor& truth,
    std::int64_t n_classes,
    std::int64_t n_clusters)
{
    const auto prepared_pred = prepare_labels(pred, "pred");
    const auto prepared_truth = prepare_labels(truth, "truth");
    if (prepared_pred.labels.sizes() != prepared_truth.labels.sizes()) {
        throw std::invalid_argument("clustering_metrics pred and truth must have the same shape");
    }
    const auto truth_one_hot = torch::one_hot(prepared_truth.labels, n_classes).to(torch::kFloat64);
    const auto pred_one_hot = torch::one_hot(prepared_pred.labels, n_clusters).to(torch::kFloat64);
    const auto confusion = torch::einsum("utc,utk->uck", {truth_one_hot, pred_one_hot}); // [U, C, K]
    return drop_unit_axis(confusion, prepared_pred.batched && prepared_truth.batched);
}

torch::Tensor cluster_purity(const torch::Tensor& confusion)
{
    const auto prepared = prepare_confusion(confusion);
    const auto& m = prepared.confusion;                // [U, C, K]
    const auto total = m.sum({1, 2});                  // [U]
    const auto dominant = std::get<0>(m.max(1)).sum(1); // sum over K of max over C -> [U]
    return drop_unit_axis(dominant / total.clamp_min(1.0), prepared.batched);
}

torch::Tensor adjusted_rand_index(const torch::Tensor& confusion)
{
    const auto prepared = prepare_confusion(confusion);
    const auto& m = prepared.confusion;   // [U, C, K]
    const auto n = m.sum({1, 2});         // [U]
    const auto a = m.sum(2);              // [U, C] true-class sizes
    const auto b = m.sum(1);              // [U, K] cluster sizes

    const auto index = comb2(m).sum({1, 2});   // [U]
    const auto sum_a = comb2(a).sum(1);        // [U]
    const auto sum_b = comb2(b).sum(1);        // [U]
    const auto expected = sum_a * sum_b / comb2(n).clamp_min(1.0);
    const auto max_index = 0.5 * (sum_a + sum_b);
    const auto denom = max_index - expected;
    // denom == 0 means trivial perfect agreement (a single block on each side).
    const auto ari = torch::where(denom.abs() > 0, (index - expected) / denom, torch::ones_like(denom));
    return drop_unit_axis(ari, prepared.batched);
}

torch::Tensor normalized_mutual_info(const torch::Tensor& confusion)
{
    const auto prepared = prepare_confusion(confusion);
    const auto& m = prepared.confusion;   // [U, C, K]
    const auto n = m.sum({1, 2}).unsqueeze(-1); // [U, 1]
    const auto a = m.sum(2);              // [U, C]
    const auto b = m.sum(1);              // [U, K]

    const auto p = m / n.unsqueeze(-1);   // [U, C, K] joint
    const auto pa = a / n;                // [U, C]
    const auto pb = b / n;                // [U, K]

    const auto outer = pa.unsqueeze(2) * pb.unsqueeze(1);        // [U, C, K] independent
    const auto mi_terms = torch::where(
        m > 0, p * (torch::log(p) - torch::log(outer)), torch::zeros_like(p));
    const auto mutual = mi_terms.sum({1, 2});                    // [U]

    const auto denom = 0.5 * (entropy(pa) + entropy(pb));        // [U]
    const auto nmi = torch::where(denom > 0, mutual / denom, torch::ones_like(denom));
    return drop_unit_axis(nmi, prepared.batched);
}

ClusteringMetrics clustering_metrics(
    const torch::Tensor& pred,
    const torch::Tensor& truth,
    std::int64_t n_classes,
    std::int64_t n_clusters)
{
    const auto prepared_pred = prepare_labels(pred, "pred");
    const bool batched = prepared_pred.batched;
    const auto confusion = confusion_matrix(pred, truth, n_classes, n_clusters);
    ClusteringMetrics metrics;
    metrics.batched = batched;
    metrics.confusion = confusion;
    metrics.purity = cluster_purity(confusion);
    metrics.adjusted_rand_index = adjusted_rand_index(confusion);
    metrics.normalized_mutual_info = normalized_mutual_info(confusion);
    return metrics;
}

namespace {

// Kuhn-Munkres (Hungarian) optimal assignment maximising total value, square n x n. Returns,
// for each row, the column it is paired with (0-indexed). O(n^3). Solves min-cost on -value.
[[nodiscard]] std::vector<std::int64_t> hungarian_max_assignment(
    const std::vector<std::vector<std::int64_t>>& value)
{
    const int n = static_cast<int>(value.size());
    constexpr std::int64_t inf = std::numeric_limits<std::int64_t>::max() / 4;
    std::vector<std::int64_t> u(n + 1, 0);
    std::vector<std::int64_t> v(n + 1, 0);
    std::vector<int> p(n + 1, 0);
    std::vector<int> way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<std::int64_t> minv(n + 1, inf);
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            std::int64_t delta = inf;
            int j1 = -1;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    const std::int64_t cur = (-value[i0 - 1][j - 1]) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<std::int64_t> column_for_row(static_cast<std::size_t>(n), 0);
    for (int j = 1; j <= n; ++j) {
        column_for_row[static_cast<std::size_t>(p[j] - 1)] = j - 1;
    }
    return column_for_row;
}

} // namespace

torch::Tensor hungarian_match(const torch::Tensor& confusion)
{
    const auto prepared = prepare_confusion(confusion); // [U, C, K]
    const auto units = prepared.confusion.size(0);
    const auto classes = prepared.confusion.size(1);
    const auto clusters = prepared.confusion.size(2);
    if (classes != clusters) {
        throw std::invalid_argument("hungarian_match requires a square confusion (n_classes == n_clusters)");
    }
    const auto counts = prepared.confusion.round().to(torch::kLong).to(torch::kCPU).contiguous();
    auto matching = torch::empty({units, classes}, torch::TensorOptions().dtype(torch::kLong));
    auto matching_acc = matching.accessor<std::int64_t, 2>();
    for (std::int64_t unit = 0; unit < units; ++unit) {
        const auto unit_counts = counts[unit];
        const auto acc = unit_counts.accessor<std::int64_t, 2>();
        std::vector<std::vector<std::int64_t>> value(
            static_cast<std::size_t>(classes), std::vector<std::int64_t>(static_cast<std::size_t>(clusters)));
        for (std::int64_t row = 0; row < classes; ++row) {
            for (std::int64_t col = 0; col < clusters; ++col) {
                value[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = acc[row][col];
            }
        }
        const auto column_for_row = hungarian_max_assignment(value);
        for (std::int64_t row = 0; row < classes; ++row) {
            matching_acc[unit][row] = column_for_row[static_cast<std::size_t>(row)];
        }
    }
    return drop_unit_axis(matching, prepared.batched);
}

torch::Tensor reorder_confusion_columns(const torch::Tensor& confusion, const torch::Tensor& matching)
{
    const auto prepared = prepare_confusion(confusion); // [U, C, K]
    auto rows_to_cluster = matching.to(torch::kLong);
    if (rows_to_cluster.dim() == 1) {
        rows_to_cluster = rows_to_cluster.unsqueeze(0);
    }
    const auto units = prepared.confusion.size(0);
    const auto classes = prepared.confusion.size(1);
    const auto index = rows_to_cluster.unsqueeze(1).expand({units, classes, rows_to_cluster.size(1)});
    const auto reordered = prepared.confusion.gather(2, index); // out[u,r,c] = M[u,r, matching[u,c]]
    return drop_unit_axis(reordered, prepared.batched);
}

MatchedClusteringScores matched_clustering_scores(const torch::Tensor& confusion)
{
    const auto prepared = prepare_confusion(confusion); // [U, C, K]
    const auto& m = prepared.confusion;
    const auto matching = hungarian_match(m); // [U, C] (m is batched)

    const auto n = m.sum({1, 2});                                       // [U]
    const auto class_size = m.sum(2);                                   // [U, C]
    const auto cluster_size = m.sum(1);                                 // [U, K]
    const auto diagonal = m.gather(2, matching.unsqueeze(-1)).squeeze(-1); // [U, C] matched overlap
    const auto matched_cluster_size = cluster_size.gather(1, matching); // [U, C]

    const auto recall = diagonal / class_size.clamp_min(1.0);
    const auto precision = diagonal / matched_cluster_size.clamp_min(1.0);
    const auto denom = (precision + recall).clamp_min(1.0e-12);
    const auto f1 = torch::where(
        precision + recall > 0, 2.0 * precision * recall / denom, torch::zeros_like(precision));
    const auto accuracy = diagonal.sum(1) / n.clamp_min(1.0);

    MatchedClusteringScores scores;
    scores.batched = prepared.batched;
    scores.matching = drop_unit_axis(matching, prepared.batched);
    scores.accuracy = drop_unit_axis(accuracy, prepared.batched);
    scores.precision = drop_unit_axis(precision, prepared.batched);
    scores.recall = drop_unit_axis(recall, prepared.batched);
    scores.f1 = drop_unit_axis(f1, prepared.batched);
    return scores;
}

} // namespace leakflow::ml
