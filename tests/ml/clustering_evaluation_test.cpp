#include "leakflow/ml/clustering_evaluation.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace {

bool expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

bool close(double value, double target, double tolerance = 1.0e-10) {
  return std::abs(value - target) <= tolerance;
}

bool metric_close(const leakflow::ml::MetricValue &metric, double target,
                  const std::string &message, double tolerance = 1.0e-10) {
  if (!metric.value.has_value()) {
    std::cerr << message << ": metric is undefined\n";
    return false;
  }
  if (!close(*metric.value, target, tolerance)) {
    std::cerr << message << ": got " << *metric.value << ", expected " << target
              << '\n';
    return false;
  }
  return true;
}

template <typename Function> bool throws_invalid_argument(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

leakflow::ml::ClusteringEvaluationUnitResult
only_unit(const leakflow::ml::ClusteringEvaluationResult &result) {
  return result.units.at(0);
}

torch::Tensor scalar_truth(const std::vector<std::int64_t> &values) {
  return torch::tensor(values, torch::kLong)
      .reshape({static_cast<std::int64_t>(values.size()), 1});
}

bool exact_metrics_equal(const leakflow::ml::ExactClusteringMetrics &left,
                         const leakflow::ml::ExactClusteringMetrics &right,
                         double tolerance = 1.0e-12) {
  const std::array<const leakflow::ml::MetricValue *, 10> left_values{{
      &left.adjusted_rand_index,
      &left.adjusted_mutual_information,
      &left.homogeneity,
      &left.completeness,
      &left.v_measure,
      &left.purity,
      &left.pair_precision,
      &left.pair_recall,
      &left.pair_f1,
      &left.normalized_mutual_information,
  }};
  const std::array<const leakflow::ml::MetricValue *, 10> right_values{{
      &right.adjusted_rand_index,
      &right.adjusted_mutual_information,
      &right.homogeneity,
      &right.completeness,
      &right.v_measure,
      &right.purity,
      &right.pair_precision,
      &right.pair_recall,
      &right.pair_f1,
      &right.normalized_mutual_information,
  }};
  for (std::size_t index = 0; index < left_values.size(); ++index) {
    const auto &lhs = *left_values[index];
    const auto &rhs = *right_values[index];
    if (lhs.metric != rhs.metric || lhs.support_count != rhs.support_count ||
        lhs.undefined_reason != rhs.undefined_reason ||
        lhs.defined() != rhs.defined()) {
      return false;
    }
    if (lhs.defined() && !close(*lhs.value, *rhs.value, tolerance)) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  using namespace leakflow::ml;

  // Stable descriptors keep generic payload/plot code independent of field-name
  // switches.
  {
    const auto descriptors = exact_clustering_metric_descriptors();
    if (!expect(descriptors.size() == 10,
                "descriptors: wrong exact metric count")) {
      return 1;
    }
    if (!expect(descriptors.front().name == "adjusted_rand_index" &&
                    descriptors.back().name == "normalized_mutual_information",
                "descriptors: names/order wrong")) {
      return 1;
    }
    const auto &purity =
        clustering_metric_descriptor(ClusteringMetricId::Purity);
    if (!expect(purity.family == MetricFamily::Exact &&
                    purity.direction == MetricDirection::HigherIsBetter &&
                    purity.averaging == MetricAveraging::Micro,
                "descriptors: purity metadata wrong")) {
      return 1;
    }
  }

  // Nontrivial sklearn-reference fixture with vector truth and sparse/negative
  // IDs.
  {
    const auto truth = torch::tensor(
        {{10, 20}, {10, 20}, {30, 40}, {30, 40}, {50, 60}, {50, 60}},
        torch::kLong);
    const auto predicted =
        torch::tensor({100, 100, -7, 42, 42, 42}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    options.dimension_names = {"left", "right"};
    const auto result = evaluate_clustering(predicted, truth, options);
    const auto &unit = only_unit(result);

    if (!expect(result.schema_version == 1 && !result.batched &&
                    result.observation_count == 6 &&
                    result.semantic_dimension_count == 2 &&
                    result.effective_options.detail == options.detail &&
                    result.effective_options.dimension_names ==
                        options.dimension_names &&
                    result.units.size() == 1,
                "reference: result envelope wrong")) {
      return 1;
    }
    if (!expect(unit.truth_group_count == 3 &&
                    unit.predicted_cluster_count == 3,
                "reference: group counts wrong")) {
      return 1;
    }
    const auto &pairs = unit.pair_counts;
    if (!expect(pairs.total_pairs == 15 && pairs.true_positive == 2 &&
                    pairs.false_positive == 2 && pairs.false_negative == 1 &&
                    pairs.true_negative == 10 &&
                    pairs.predicted_within_cluster_pairs == 4 &&
                    pairs.true_within_group_pairs == 3,
                "reference: pair counts wrong")) {
      return 1;
    }

    const auto &exact = unit.exact;
    if (!metric_close(exact.adjusted_rand_index, 0.44444444444444442,
                      "reference: ARI")) {
      return 1;
    }
    if (!metric_close(exact.adjusted_mutual_information, 0.50236070272027378,
                      "reference: AMI", 1.0e-9)) {
      return 1;
    }
    if (!metric_close(exact.homogeneity, 0.71030991785715247,
                      "reference: homogeneity")) {
      return 1;
    }
    if (!metric_close(exact.completeness, 0.77155617367947116,
                      "reference: completeness")) {
      return 1;
    }
    if (!metric_close(exact.v_measure, 0.73966737680075900,
                      "reference: V-measure")) {
      return 1;
    }
    if (!metric_close(exact.normalized_mutual_information, 0.73966737680075922,
                      "reference: NMI")) {
      return 1;
    }
    if (!metric_close(exact.purity, 5.0 / 6.0, "reference: purity")) {
      return 1;
    }
    if (!metric_close(exact.pair_precision, 0.5, "reference: pair precision") ||
        !metric_close(exact.pair_recall, 2.0 / 3.0, "reference: pair recall") ||
        !metric_close(exact.pair_f1, 4.0 / 7.0, "reference: pair F1")) {
      return 1;
    }
    if (!expect(exact.pair_precision.support_count == 4 &&
                    exact.pair_recall.support_count == 3 &&
                    exact.pair_f1.support_count == 7,
                "reference: pair supports wrong")) {
      return 1;
    }

    if (!expect(unit.partition_detail.has_value(),
                "reference: full detail missing")) {
      return 1;
    }
    const auto &detail = *unit.partition_detail;
    if (!expect(torch::equal(detail.predicted_ids,
                             torch::tensor({-7, 42, 100}, torch::kLong)),
                "reference: predicted ID normalization wrong")) {
      return 1;
    }
    if (!expect(torch::equal(detail.truth_vectors,
                             torch::tensor({{10, 20}, {30, 40}, {50, 60}},
                                           torch::kLong)),
                "reference: truth-vector normalization wrong")) {
      return 1;
    }
    if (!expect(torch::equal(detail.contingency.truth_group_indices,
                             torch::tensor({0, 1, 1, 2}, torch::kLong)) &&
                    torch::equal(detail.contingency.predicted_cluster_indices,
                                 torch::tensor({2, 0, 1, 1}, torch::kLong)) &&
                    torch::equal(detail.contingency.counts,
                                 torch::tensor({2, 1, 1, 2}, torch::kLong)),
                "reference: sparse contingency wrong")) {
      return 1;
    }

    // Observation order must not change normalized detail or scores.
    const auto permutation = torch::tensor({5, 0, 3, 1, 4, 2}, torch::kLong);
    const auto permuted =
        evaluate_clustering(predicted.index_select(0, permutation),
                            truth.index_select(0, permutation), options);
    const auto &permuted_unit = only_unit(permuted);
    if (!expect(exact_metrics_equal(unit.exact, permuted_unit.exact),
                "reference: observation permutation changed metrics")) {
      return 1;
    }
    if (!expect(
            torch::equal(detail.predicted_ids,
                         permuted_unit.partition_detail->predicted_ids) &&
                torch::equal(detail.truth_vectors,
                             permuted_unit.partition_detail->truth_vectors) &&
                torch::equal(detail.contingency.truth_group_indices,
                             permuted_unit.partition_detail->contingency
                                 .truth_group_indices) &&
                torch::equal(detail.contingency.predicted_cluster_indices,
                             permuted_unit.partition_detail->contingency
                                 .predicted_cluster_indices) &&
                torch::equal(
                    detail.contingency.counts,
                    permuted_unit.partition_detail->contingency.counts),
            "reference: observation permutation changed normalized detail")) {
      return 1;
    }
  }

  // Symmetric scores are invariant when truth and prediction are exchanged;
  // homogeneity and completeness exchange roles.
  {
    const auto truth_labels = std::vector<std::int64_t>{0, 0, 1, 1, 2, 2};
    const auto predicted_labels =
        std::vector<std::int64_t>{100, 100, -7, 42, 42, 42};
    const auto &forward =
        only_unit(evaluate_clustering(torch::tensor(predicted_labels),
                                      scalar_truth(truth_labels)))
            .exact;
    const auto &reverse =
        only_unit(evaluate_clustering(torch::tensor(truth_labels),
                                      scalar_truth(predicted_labels)))
            .exact;
    if (!expect(
            close(*forward.adjusted_rand_index.value,
                  *reverse.adjusted_rand_index.value) &&
                close(*forward.adjusted_mutual_information.value,
                      *reverse.adjusted_mutual_information.value, 1.0e-9) &&
                close(*forward.v_measure.value, *reverse.v_measure.value) &&
                close(*forward.normalized_mutual_information.value,
                      *reverse.normalized_mutual_information.value) &&
                close(*forward.homogeneity.value,
                      *reverse.completeness.value) &&
                close(*forward.completeness.value, *reverse.homogeneity.value),
            "symmetry: exchanged partitions changed exact metrics")) {
      return 1;
    }
  }

  // A non-degenerate perfect vector partition exercises expected MI instead
  // of the all-singleton/all-constant shortcuts.
  {
    const auto truth = torch::tensor(
        {{1, 10}, {1, 10}, {2, 20}, {2, 20}, {3, 30}, {3, 30}}, torch::kLong);
    const auto predicted = torch::tensor({7, 7, -3, -3, 99, 99}, torch::kLong);
    const auto &unit = only_unit(evaluate_clustering(predicted, truth));
    const auto &exact = unit.exact;
    if (!metric_close(exact.adjusted_rand_index, 1.0, "perfect: ARI") ||
        !metric_close(exact.adjusted_mutual_information, 1.0, "perfect: AMI",
                      1.0e-9) ||
        !metric_close(exact.homogeneity, 1.0, "perfect: homogeneity") ||
        !metric_close(exact.completeness, 1.0, "perfect: completeness") ||
        !metric_close(exact.v_measure, 1.0, "perfect: V") ||
        !metric_close(exact.normalized_mutual_information, 1.0,
                      "perfect: NMI") ||
        !metric_close(exact.purity, 1.0, "perfect: purity") ||
        !metric_close(exact.pair_precision, 1.0, "perfect: pair precision") ||
        !metric_close(exact.pair_recall, 1.0, "perfect: pair recall") ||
        !metric_close(exact.pair_f1, 1.0, "perfect: pair F1")) {
      return 1;
    }
    if (!expect(unit.pair_counts.total_pairs == 15 &&
                    unit.pair_counts.true_positive == 3 &&
                    unit.pair_counts.false_positive == 0 &&
                    unit.pair_counts.false_negative == 0 &&
                    unit.pair_counts.true_negative == 12,
                "perfect: unordered pair counts wrong")) {
      return 1;
    }
  }

  // Direct AMI reference contingency from scikit-learn's expected-MI path.
  {
    const auto truth = scalar_truth({
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
    });
    const auto predicted = torch::tensor(
        {
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            2,
            2,
            2,
        },
        torch::kLong);
    const auto &exact = only_unit(evaluate_clustering(predicted, truth)).exact;
    if (!metric_close(exact.adjusted_rand_index, 0.2669404517453799,
                      "AMI fixture: ARI") ||
        !metric_close(exact.adjusted_mutual_information, 0.2782096465726979,
                      "AMI fixture: AMI", 1.0e-9) ||
        !metric_close(exact.homogeneity, 0.3746022423885998,
                      "AMI fixture: homogeneity") ||
        !metric_close(exact.completeness, 0.38217335812507564,
                      "AMI fixture: completeness") ||
        !metric_close(exact.v_measure, 0.3783499278720408, "AMI fixture: V") ||
        !metric_close(exact.normalized_mutual_information, 0.37834992787204075,
                      "AMI fixture: NMI")) {
      return 1;
    }
  }

  // Pair-score undefined semantics and sklearn-compatible degenerate
  // partitions.
  {
    const auto pure_split = only_unit(evaluate_clustering(
        torch::tensor({0, 1, 2, 3}, torch::kLong), scalar_truth({0, 0, 1, 1})));
    if (!metric_close(pure_split.exact.adjusted_rand_index, 0.0,
                      "split: ARI") ||
        !metric_close(pure_split.exact.adjusted_mutual_information, 0.0,
                      "split: AMI", 1.0e-9) ||
        !metric_close(pure_split.exact.homogeneity, 1.0,
                      "split: homogeneity") ||
        !metric_close(pure_split.exact.completeness, 0.5,
                      "split: completeness") ||
        !metric_close(pure_split.exact.v_measure, 2.0 / 3.0, "split: V") ||
        !metric_close(pure_split.exact.purity, 1.0, "split: purity")) {
      return 1;
    }
    if (!expect(!pure_split.exact.pair_precision.defined() &&
                    pure_split.exact.pair_precision.undefined_reason ==
                        MetricUndefinedReason::NoPredictedWithinClusterPairs &&
                    metric_close(pure_split.exact.pair_recall, 0.0,
                                 "split: pair recall") &&
                    !pure_split.exact.pair_f1.defined() &&
                    pure_split.exact.pair_f1.undefined_reason ==
                        MetricUndefinedReason::DependentMetricUndefined,
                "split: undefined pair semantics wrong")) {
      return 1;
    }

    const auto total_merge = only_unit(evaluate_clustering(
        torch::zeros({4}, torch::kLong), scalar_truth({0, 0, 1, 1})));
    if (!metric_close(total_merge.exact.pair_precision, 1.0 / 3.0,
                      "merge: pair precision") ||
        !metric_close(total_merge.exact.pair_recall, 1.0,
                      "merge: pair recall") ||
        !metric_close(total_merge.exact.pair_f1, 0.5, "merge: pair F1") ||
        !metric_close(total_merge.exact.completeness, 1.0,
                      "merge: completeness") ||
        !metric_close(total_merge.exact.homogeneity, 0.0,
                      "merge: homogeneity")) {
      return 1;
    }

    const auto crossed = only_unit(evaluate_clustering(
        torch::tensor({0, 1, 0, 1}, torch::kLong), scalar_truth({0, 0, 1, 1})));
    if (!metric_close(crossed.exact.adjusted_rand_index, -0.5,
                      "crossed: ARI") ||
        !metric_close(crossed.exact.adjusted_mutual_information, -0.5,
                      "crossed: AMI", 1.0e-9) ||
        !metric_close(crossed.exact.pair_precision, 0.0,
                      "crossed: pair precision") ||
        !metric_close(crossed.exact.pair_recall, 0.0, "crossed: pair recall") ||
        !metric_close(crossed.exact.pair_f1, 0.0, "crossed: pair F1")) {
      return 1;
    }

    const auto singleton = only_unit(evaluate_clustering(
        torch::tensor({-9}, torch::kLong), scalar_truth({42})));
    if (!metric_close(singleton.exact.adjusted_rand_index, 1.0,
                      "singleton: ARI") ||
        !metric_close(singleton.exact.adjusted_mutual_information, 1.0,
                      "singleton: AMI") ||
        !metric_close(singleton.exact.v_measure, 1.0, "singleton: V") ||
        !expect(!singleton.exact.pair_precision.defined() &&
                    !singleton.exact.pair_recall.defined() &&
                    !singleton.exact.pair_f1.defined(),
                "singleton: pair metrics should be undefined")) {
      return 1;
    }

    const auto all_singletons = only_unit(evaluate_clustering(
        torch::tensor({9, 8, 7, 6}, torch::kLong), scalar_truth({0, 1, 2, 3})));
    if (!metric_close(all_singletons.exact.adjusted_rand_index, 1.0,
                      "all-singleton: ARI") ||
        !metric_close(all_singletons.exact.adjusted_mutual_information, 1.0,
                      "all-singleton: AMI") ||
        !metric_close(all_singletons.exact.normalized_mutual_information, 1.0,
                      "all-singleton: NMI")) {
      return 1;
    }

    const auto both_constant = only_unit(evaluate_clustering(
        torch::full({4}, 9, torch::kLong), scalar_truth({0, 0, 0, 0})));
    if (!metric_close(both_constant.exact.adjusted_rand_index, 1.0,
                      "both-constant: ARI") ||
        !metric_close(both_constant.exact.adjusted_mutual_information, 1.0,
                      "both-constant: AMI") ||
        !metric_close(both_constant.exact.homogeneity, 1.0,
                      "both-constant: homogeneity") ||
        !metric_close(both_constant.exact.completeness, 1.0,
                      "both-constant: completeness") ||
        !metric_close(both_constant.exact.v_measure, 1.0, "both-constant: V") ||
        !metric_close(both_constant.exact.normalized_mutual_information, 1.0,
                      "both-constant: NMI") ||
        !metric_close(both_constant.exact.pair_f1, 1.0,
                      "both-constant: pair F1")) {
      return 1;
    }

    const auto constant_truth = only_unit(evaluate_clustering(
        torch::tensor({0, 1, 2, 3}, torch::kLong), scalar_truth({8, 8, 8, 8})));
    if (!metric_close(constant_truth.exact.adjusted_rand_index, 0.0,
                      "constant-truth: ARI") ||
        !metric_close(constant_truth.exact.adjusted_mutual_information, 0.0,
                      "constant-truth: AMI") ||
        !metric_close(constant_truth.exact.homogeneity, 1.0,
                      "constant-truth: homogeneity") ||
        !metric_close(constant_truth.exact.completeness, 0.0,
                      "constant-truth: completeness") ||
        !metric_close(constant_truth.exact.v_measure, 0.0,
                      "constant-truth: V") ||
        !metric_close(constant_truth.exact.normalized_mutual_information, 0.0,
                      "constant-truth: NMI")) {
      return 1;
    }

    const auto constant_prediction = only_unit(evaluate_clustering(
        torch::zeros({4}, torch::kLong), scalar_truth({0, 1, 2, 3})));
    if (!metric_close(constant_prediction.exact.adjusted_rand_index, 0.0,
                      "constant-prediction: ARI") ||
        !metric_close(constant_prediction.exact.adjusted_mutual_information,
                      0.0, "constant-prediction: AMI") ||
        !metric_close(constant_prediction.exact.homogeneity, 0.0,
                      "constant-prediction: homogeneity") ||
        !metric_close(constant_prediction.exact.completeness, 1.0,
                      "constant-prediction: completeness") ||
        !metric_close(constant_prediction.exact.v_measure, 0.0,
                      "constant-prediction: V") ||
        !metric_close(constant_prediction.exact.normalized_mutual_information,
                      0.0, "constant-prediction: NMI")) {
      return 1;
    }
  }

  // Leading units are evaluated independently and may have different G and K.
  {
    const auto truth0 = torch::tensor(
        {{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}, torch::kLong);
    const auto truth1 = torch::tensor(
        {{5, 1, 2, 3}, {5, 1, 2, 3}, {5, 1, 2, 3}, {6, 1, 2, 3}}, torch::kLong);
    const auto predicted = torch::stack({
        torch::tensor({9, 9, -2, -2}, torch::kLong),
        torch::tensor({0, 1, 2, 2}, torch::kLong),
    });
    const auto result =
        evaluate_clustering(predicted, torch::stack({truth0, truth1}));
    if (!expect(
            result.batched && result.units.size() == 2 &&
                result.semantic_dimension_count == 4 &&
                result.effective_options.detail ==
                    ClusteringEvaluationDetail::Global &&
                result.effective_options.dimension_names ==
                    std::vector<std::string>({"dimension_0", "dimension_1",
                                              "dimension_2", "dimension_3"}),
            "batched: envelope/default dimension names wrong")) {
      return 1;
    }
    if (!expect(result.units[0].truth_group_count == 2 &&
                    result.units[0].predicted_cluster_count == 2 &&
                    result.units[1].truth_group_count == 2 &&
                    result.units[1].predicted_cluster_count == 3,
                "batched: ragged group counts wrong")) {
      return 1;
    }
    if (!metric_close(result.units[0].exact.adjusted_rand_index, 1.0,
                      "batched: unit 0 ARI")) {
      return 1;
    }
    if (!expect(!result.units[0].partition_detail.has_value() &&
                    !result.units[1].partition_detail.has_value(),
                "batched: global detail unexpectedly materialized partition "
                "data")) {
      return 1;
    }
  }

  // Semantic values retain exact stored equality and full-detail dtype.
  {
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    const std::int64_t large0 = 9'007'199'254'740'992LL;
    const std::int64_t large1 = 9'007'199'254'740'993LL;
    const auto integral = evaluate_clustering(
        torch::tensor({0, 0, 1}, torch::kLong),
        torch::tensor({large0, large0, large1}, torch::kLong).reshape({3, 1}),
        options);
    if (!expect(only_unit(integral).truth_group_count == 2 &&
                    only_unit(integral)
                            .partition_detail->truth_vectors.scalar_type() ==
                        torch::kLong,
                "dtype: integral truth was merged or converted")) {
      return 1;
    }

    const auto base = 0.1;
    const auto next = std::nextafter(base, 1.0);
    const auto floating = evaluate_clustering(
        torch::tensor({0, 0, 1}, torch::kLong),
        torch::tensor({base, base, next}, torch::kFloat64).reshape({3, 1}),
        options);
    if (!expect(only_unit(floating).truth_group_count == 2 &&
                    only_unit(floating)
                            .partition_detail->truth_vectors.scalar_type() ==
                        torch::kFloat64,
                "dtype: floating truth was merged or converted")) {
      return 1;
    }

    const auto byte_truth =
        torch::tensor({1, 1, 9}, torch::kUInt8).reshape({3, 1});
    const auto byte_result = evaluate_clustering(
        torch::tensor({0, 0, 1}, torch::kLong), byte_truth, options);
    if (!expect(only_unit(byte_result)
                        .partition_detail->truth_vectors.scalar_type() ==
                    torch::kUInt8,
                "dtype: byte truth dtype was not retained")) {
      return 1;
    }
  }

  // Strict input validation: no silent rank or dtype coercion.
  {
    const auto valid_predicted = torch::tensor({0, 1}, torch::kLong);
    const auto valid_truth = scalar_truth({0, 1});
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      torch::tensor({0.0, 1.0}), valid_truth));
                }),
                "validation: floating predicted labels accepted")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted,
                      torch::zeros({2, 1}, torch::kComplexDouble)));
                }),
                "validation: complex semantic truth accepted")) {
      return 1;
    }
    if (!expect(
            throws_invalid_argument([&] {
              static_cast<void>(evaluate_clustering(
                  valid_predicted,
                  torch::tensor({0.0, std::numeric_limits<double>::quiet_NaN()})
                      .reshape({2, 1})));
            }),
            "validation: NaN semantic truth accepted")) {
      return 1;
    }
    if (!expect(
            throws_invalid_argument([&] {
              static_cast<void>(evaluate_clustering(
                  valid_predicted,
                  torch::tensor({0.0, std::numeric_limits<double>::infinity()})
                      .reshape({2, 1})));
            }),
            "validation: infinite semantic truth accepted")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted.unsqueeze(0), valid_truth));
                }),
                "validation: mixed batched/unbatched ranks accepted")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, torch::zeros({3, 1}, torch::kLong)));
                }),
                "validation: mismatched observations accepted")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(
                      evaluate_clustering(torch::empty({0}, torch::kLong),
                                          torch::empty({0, 1}, torch::kLong)));
                }),
                "validation: empty observations accepted")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, torch::empty({2, 0}, torch::kLong)));
                }),
                "validation: empty semantic dimension accepted")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      torch::empty({0, 2}, torch::kLong),
                      torch::empty({0, 2, 1}, torch::kLong)));
                }),
                "validation: empty unit axis accepted")) {
      return 1;
    }

    auto wide_predicted = torch::empty(
        {2}, torch::TensorOptions().dtype(c10::ScalarType::UInt64));
    wide_predicted.data_ptr<std::uint64_t>()[0] = 0;
    wide_predicted.data_ptr<std::uint64_t>()[1] =
        std::numeric_limits<std::uint64_t>::max();
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(
                      evaluate_clustering(wide_predicted, valid_truth));
                }),
                "validation: out-of-range uint64 predicted label accepted")) {
      return 1;
    }

    auto wide_truth = torch::empty(
        {2, 1}, torch::TensorOptions().dtype(c10::ScalarType::UInt64));
    wide_truth.data_ptr<std::uint64_t>()[0] = 0;
    wide_truth.data_ptr<std::uint64_t>()[1] =
        std::numeric_limits<std::uint64_t>::max();
    ClusteringEvaluationOptions wide_truth_options;
    wide_truth_options.detail = ClusteringEvaluationDetail::Full;
    const auto wide_truth_result =
        evaluate_clustering(valid_predicted, wide_truth, wide_truth_options);
    if (!expect(only_unit(wide_truth_result).truth_group_count == 2 &&
                    torch::equal(only_unit(wide_truth_result)
                                     .partition_detail->truth_vectors,
                                 wide_truth),
                "validation: uint64 semantic truth was not preserved")) {
      return 1;
    }

    ClusteringEvaluationOptions wrong_count;
    wrong_count.dimension_names = {"one", "two"};
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, wrong_count));
                }),
                "validation: dimension-name count mismatch accepted")) {
      return 1;
    }
    ClusteringEvaluationOptions empty_name;
    empty_name.dimension_names = {""};
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, empty_name));
                }),
                "validation: empty dimension name accepted")) {
      return 1;
    }
    ClusteringEvaluationOptions duplicate_names;
    duplicate_names.dimension_names = {"same", "same"};
    const auto two_dim_truth = torch::tensor({{0, 1}, {1, 0}}, torch::kLong);
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, two_dim_truth, duplicate_names));
                }),
                "validation: duplicate dimension names accepted")) {
      return 1;
    }
  }

  // Sparse high-cardinality case: Z grows with N, not with G*K.
  {
    constexpr std::int64_t observations = 5'000;
    const auto truth =
        torch::arange(observations, torch::kLong).reshape({observations, 1});
    const auto predicted = torch::arange(observations, torch::kLong) * 17 - 9;
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    const auto result = evaluate_clustering(predicted, truth, options);
    const auto &unit = only_unit(result);
    if (!expect(unit.truth_group_count == observations &&
                    unit.predicted_cluster_count == observations &&
                    unit.partition_detail->contingency.counts.numel() ==
                        observations,
                "sparse stress: detail is not sparse")) {
      return 1;
    }
    if (!metric_close(unit.exact.adjusted_rand_index, 1.0,
                      "sparse stress: ARI") ||
        !metric_close(unit.exact.adjusted_mutual_information, 1.0,
                      "sparse stress: AMI")) {
      return 1;
    }
  }

  // Repeated marginal sizes must not trigger a raw G*K expected-MI loop. This
  // crossed ring has many groups, does not take an AMI degeneracy shortcut,
  // and has only one distinct marginal-size pair.
  {
    constexpr std::int64_t observations = 5'000;
    auto truth_labels = torch::empty({observations}, torch::kLong);
    auto predicted = torch::empty({observations}, torch::kLong);
    auto truth_access = truth_labels.accessor<std::int64_t, 1>();
    auto predicted_access = predicted.accessor<std::int64_t, 1>();
    for (std::int64_t observation = 0; observation < observations;
         ++observation) {
      truth_access[observation] = observation / 2;
      predicted_access[observation] = ((observation + 1) % observations) / 2;
    }

    const auto &unit = only_unit(evaluate_clustering(
        predicted, truth_labels.reshape({observations, 1})));
    if (!expect(
            unit.truth_group_count == observations / 2 &&
                unit.predicted_cluster_count == observations / 2 &&
                unit.pair_counts.true_positive == 0 &&
                unit.pair_counts.false_positive == observations / 2 &&
                unit.pair_counts.false_negative == observations / 2 &&
                unit.exact.adjusted_mutual_information.defined() &&
                std::isfinite(*unit.exact.adjusted_mutual_information.value),
            "expected-MI stress: repeated marginals were mishandled")) {
      return 1;
    }
  }

  // An all-singleton truth partition against one predicted doublet is another
  // non-short-circuited AMI case: MI equals its fixed-margin expectation, so
  // the adjusted score is zero even at very high cardinality.
  {
    constexpr std::int64_t observations = 5'000;
    const auto truth_labels = torch::arange(observations, torch::kLong);
    auto predicted = torch::arange(observations, torch::kLong);
    predicted.data_ptr<std::int64_t>()[1] = 0;
    const auto &unit = only_unit(evaluate_clustering(
        predicted, truth_labels.reshape({observations, 1})));
    if (!expect(unit.truth_group_count == observations &&
                    unit.predicted_cluster_count == observations - 1 &&
                    unit.pair_counts.true_positive == 0 &&
                    unit.pair_counts.false_positive == 1 &&
                    unit.pair_counts.false_negative == 0,
                "near-singleton stress: partition counts wrong") ||
        !metric_close(unit.exact.adjusted_mutual_information, 0.0,
                      "near-singleton stress: AMI", 1.0e-10)) {
      return 1;
    }
  }

  std::cout << "clustering_evaluation tests passed\n";
  return 0;
}
