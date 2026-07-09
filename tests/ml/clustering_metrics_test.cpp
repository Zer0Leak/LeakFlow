#include "leakflow/ml/clustering_metrics.hpp"

#include <cstdint>
#include <iostream>
#include <torch/torch.h>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

bool close(double value, double target, double tol = 1.0e-9)
{
    return std::abs(value - target) < tol;
}

} // namespace

int main()
{
    // --- Confusion matrix counts ---
    {
        const auto truth = torch::tensor({0, 0, 1, 1}, torch::kLong);
        const auto pred = torch::tensor({1, 1, 0, 0}, torch::kLong); // clusters are a permutation
        const auto m = leakflow::ml::confusion_matrix(pred, truth, /*n_classes=*/2, /*n_clusters=*/2);
        const auto expected = torch::tensor({{0.0, 2.0}, {2.0, 0.0}}, torch::kFloat64);
        if (!expect(torch::equal(m, expected), "confusion matrix counts wrong")) {
            return 1;
        }
    }

    // --- Perfect clustering (identity and a permutation both score 1) ---
    {
        const auto truth = torch::tensor({0, 0, 1, 1, 2, 2}, torch::kLong);
        const auto perm = torch::tensor({2, 2, 0, 0, 1, 1}, torch::kLong); // relabelled but identical partition
        const auto metrics = leakflow::ml::clustering_metrics(perm, truth, 3, 3);
        if (!expect(close(metrics.purity.item<double>(), 1.0), "perfect: purity != 1")) {
            return 1;
        }
        if (!expect(close(metrics.adjusted_rand_index.item<double>(), 1.0), "perfect: ARI != 1")) {
            return 1;
        }
        if (!expect(close(metrics.normalized_mutual_info.item<double>(), 1.0), "perfect: NMI != 1")) {
            return 1;
        }
    }

    // --- Single predicted cluster: ARI = 0, NMI = 0, purity = biggest-class fraction ---
    {
        const auto truth = torch::tensor({0, 0, 0, 1, 1, 2}, torch::kLong);
        const auto pred = torch::zeros({6}, torch::kLong);
        const auto metrics = leakflow::ml::clustering_metrics(pred, truth, 3, 1);
        if (!expect(close(metrics.purity.item<double>(), 0.5), "single-cluster: purity != 0.5")) {
            return 1;
        }
        if (!expect(close(metrics.adjusted_rand_index.item<double>(), 0.0, 1.0e-9), "single-cluster: ARI != 0")) {
            return 1;
        }
        if (!expect(close(metrics.normalized_mutual_info.item<double>(), 0.0, 1.0e-9), "single-cluster: NMI != 0")) {
            return 1;
        }
    }

    // --- Batched over U: unit 0 perfect, unit 1 single-cluster ---
    {
        const auto truth = torch::tensor({{0, 0, 1, 1}, {0, 0, 1, 1}}, torch::kLong);
        const auto pred = torch::tensor({{1, 1, 0, 0}, {0, 0, 0, 0}}, torch::kLong);
        const auto metrics = leakflow::ml::clustering_metrics(pred, truth, 2, 2);
        if (!expect(metrics.batched && metrics.adjusted_rand_index.sizes() == torch::IntArrayRef({2}),
                    "batched: shape wrong")) {
            return 1;
        }
        if (!expect(close(metrics.adjusted_rand_index[0].item<double>(), 1.0)
                        && close(metrics.adjusted_rand_index[1].item<double>(), 0.0, 1.0e-9),
                    "batched: per-unit ARI wrong")) {
            return 1;
        }
        if (!expect(close(metrics.purity[0].item<double>(), 1.0) && close(metrics.purity[1].item<double>(), 0.5),
                    "batched: per-unit purity wrong")) {
            return 1;
        }
    }

    // --- Hungarian matching resolves the collision example, gives accuracy ---
    {
        // rows = class (A,B,C), cols = cluster; A and B both peak toward cluster 0/1.
        const auto confusion = torch::tensor(
            {{50.0, 45.0, 5.0}, {20.0, 30.0, 10.0}, {5.0, 8.0, 30.0}}, torch::kFloat64);
        const auto match = leakflow::ml::hungarian_match(confusion);
        // Optimal bijection is A->0, B->1, C->2 (total overlap 110, N=203).
        if (!expect(torch::equal(match, torch::tensor({0, 1, 2}, torch::kLong)), "hungarian: collision match wrong")) {
            return 1;
        }
        const auto scores = leakflow::ml::matched_clustering_scores(confusion);
        if (!expect(close(scores.accuracy.item<double>(), 110.0 / 203.0), "hungarian: accuracy wrong")) {
            return 1;
        }
    }

    // --- Permutation: matching recovers it; accuracy/precision/recall/F1 = 1; reorder diagonalises ---
    {
        const auto confusion = torch::tensor(
            {{0.0, 0.0, 10.0}, {10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}}, torch::kFloat64);
        const auto scores = leakflow::ml::matched_clustering_scores(confusion);
        if (!expect(torch::equal(scores.matching, torch::tensor({2, 0, 1}, torch::kLong)), "match: permutation wrong")) {
            return 1;
        }
        if (!expect(close(scores.accuracy.item<double>(), 1.0), "match: accuracy != 1")) {
            return 1;
        }
        const auto ones = torch::ones({3}, torch::kFloat64);
        if (!expect(torch::allclose(scores.precision, ones) && torch::allclose(scores.recall, ones)
                        && torch::allclose(scores.f1, ones),
                    "match: precision/recall/f1 != 1")) {
            return 1;
        }
        const auto reordered = leakflow::ml::reorder_confusion_columns(confusion, scores.matching);
        if (!expect(torch::equal(reordered, 10.0 * torch::eye(3, torch::kFloat64)), "reorder: not diagonalised")) {
            return 1;
        }
    }

    std::cout << "clustering_metrics tests passed\n";
    return 0;
}
