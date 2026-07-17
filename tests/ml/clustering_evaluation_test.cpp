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

bool metric_undefined(const leakflow::ml::MetricValue &metric,
                      leakflow::ml::MetricUndefinedReason reason,
                      std::uint64_t support, const std::string &message) {
  if (metric.defined() || metric.undefined_reason != reason ||
      metric.support_count != support) {
    std::cerr << message << ": undefined state mismatch\n";
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
    const auto exact_descriptors = exact_clustering_metric_descriptors();
    const auto descriptors = clustering_metric_descriptors();
    if (!expect(exact_descriptors.size() == 10 && descriptors.size() == 30,
                "descriptors: wrong exact metric count")) {
      return 1;
    }
    if (!expect(exact_descriptors.front().name == "adjusted_rand_index" &&
                    exact_descriptors.back().name ==
                        "normalized_mutual_information" &&
                    descriptors.back().name == "semantic_partition_quality",
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
    const auto &semantic = clustering_metric_descriptor(
        ClusteringMetricId::SemanticImpurityDimensionMicro);
    const auto &fragmentation =
        clustering_metric_descriptor(ClusteringMetricId::FragmentationMacro);
    const auto &alignment = clustering_metric_descriptor(
        ClusteringMetricId::ExactAlignmentJaccardPerGroup);
    const auto &combined =
        clustering_metric_descriptor(ClusteringMetricId::CombinedQuality);
    const auto &partition_separation = clustering_metric_descriptor(
        ClusteringMetricId::SemanticPartitionSeparation);
    const auto &partition_quality = clustering_metric_descriptor(
        ClusteringMetricId::SemanticPartitionQuality);
    if (!expect(semantic.family == MetricFamily::Semantic &&
                    semantic.direction == MetricDirection::LowerIsBetter &&
                    semantic.averaging == MetricAveraging::PerDimension &&
                    fragmentation.family == MetricFamily::Fragmentation &&
                    fragmentation.direction == MetricDirection::LowerIsBetter &&
                    fragmentation.averaging == MetricAveraging::Macro &&
                    alignment.family == MetricFamily::Alignment &&
                    alignment.direction == MetricDirection::HigherIsBetter &&
                    alignment.averaging == MetricAveraging::PerGroup &&
                    combined.family == MetricFamily::Combined &&
                    combined.direction == MetricDirection::HigherIsBetter &&
                    combined.averaging == MetricAveraging::Micro &&
                    partition_separation.family == MetricFamily::Semantic &&
                    partition_separation.direction ==
                        MetricDirection::HigherIsBetter &&
                    partition_quality.family == MetricFamily::Combined &&
                    partition_quality.direction ==
                        MetricDirection::HigherIsBetter,
                "descriptors: metric metadata wrong")) {
      return 1;
    }
    if (!expect(metric_family_name(MetricFamily::Exact) == "exact" &&
                    metric_family_name(MetricFamily::Semantic) == "semantic" &&
                    metric_family_name(MetricFamily::Fragmentation) ==
                        "fragmentation" &&
                    metric_family_name(MetricFamily::Alignment) ==
                        "alignment" &&
                    metric_family_name(MetricFamily::Combined) == "combined" &&
                    metric_direction_name(MetricDirection::HigherIsBetter) ==
                        "higher_is_better" &&
                    metric_direction_name(MetricDirection::LowerIsBetter) ==
                        "lower_is_better" &&
                    metric_averaging_name(MetricAveraging::None) == "none" &&
                    metric_averaging_name(MetricAveraging::Micro) == "micro" &&
                    metric_averaging_name(MetricAveraging::Macro) == "macro" &&
                    metric_averaging_name(MetricAveraging::PerCluster) ==
                        "per_cluster" &&
                    metric_averaging_name(MetricAveraging::PerGroup) ==
                        "per_group" &&
                    metric_averaging_name(MetricAveraging::PerDimension) ==
                        "per_dimension" &&
                    metric_undefined_reason_name(MetricUndefinedReason::None) ==
                        "none" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::NoPredictedWithinClusterPairs) ==
                        "no_predicted_within_cluster_pairs" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::NoTrueWithinGroupPairs) ==
                        "no_true_within_group_pairs" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::DependentMetricUndefined) ==
                        "dependent_metric_undefined" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::SemanticDisabled) ==
                        "semantic_disabled" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::NoEligiblePredictedClusters) ==
                        "no_eligible_predicted_clusters" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::NoMergeErrorPairs) ==
                        "no_merge_error_pairs" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::NoEligibleTruthGroups) ==
                        "no_eligible_truth_groups" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::NoMatchedPredictedCluster) ==
                        "no_matched_predicted_cluster" &&
                    metric_undefined_reason_name(
                        MetricUndefinedReason::NoSemanticVariation) ==
                        "no_semantic_variation",
                "descriptors: enum names wrong")) {
      return 1;
    }
    if (!expect(throws_invalid_argument([] {
                  static_cast<void>(
                      metric_family_name(static_cast<MetricFamily>(99)));
                }) &&
                    throws_invalid_argument([] {
                      static_cast<void>(metric_direction_name(
                          static_cast<MetricDirection>(99)));
                    }) &&
                    throws_invalid_argument([] {
                      static_cast<void>(metric_averaging_name(
                          static_cast<MetricAveraging>(99)));
                    }) &&
                    throws_invalid_argument([] {
                      static_cast<void>(metric_undefined_reason_name(
                          static_cast<MetricUndefinedReason>(99)));
                    }),
                "descriptors: invalid enum names accepted")) {
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

    if (!expect(result.schema_version == 5 && !result.batched &&
                    result.observation_count == 6 &&
                    result.semantic_dimension_count == 2 &&
                    result.effective_options.detail == options.detail &&
                    result.effective_options.dimension_names ==
                        options.dimension_names &&
                    result.effective_options.alignment ==
                        AlignmentEvaluationMode::None &&
                    !result.effective_options.combined_quality &&
                    !result.effective_options.semantic_partition_quality &&
                    !unit.combined_quality.has_value() &&
                    !unit.semantic_partition_quality.has_value() &&
                    !unit.alignment_identities.has_value() &&
                    !unit.exact_alignment.has_value() &&
                    !unit.semantic_alignment.has_value() &&
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

  // A2 mixed semantic fixture: merge impurity and fragmentation differ, and
  // both p=1 and p=2 have hand-computed reference values.
  {
    const auto truth = torch::tensor(
        {{0, 0}, {0, 0}, {0, 0}, {1, 0}, {1, 0}, {0, 2}}, torch::kLong);
    const auto predicted =
        torch::tensor({10, 10, 20, 10, 20, 10}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {2.0, 4.0};
    options.semantic_weights = {1.0, 1.0};
    options.power = 1;

    const auto result = evaluate_clustering(predicted, truth, options);
    const auto &unit = only_unit(result);
    const auto &semantic = unit.semantic;
    if (!expect(result.effective_options.semantic ==
                        SemanticEvaluationMode::Power &&
                    result.effective_options.semantic_ranges ==
                        options.semantic_ranges &&
                    result.effective_options.semantic_weights ==
                        options.semantic_weights &&
                    result.effective_options.power == 1,
                "A2 p1: effective semantic options wrong") ||
        !expect(unit.pair_counts.true_positive == 1 &&
                    unit.pair_counts.false_positive == 6 &&
                    unit.pair_counts.false_negative == 3 &&
                    unit.pair_counts.predicted_within_cluster_pairs == 7 &&
                    unit.pair_counts.true_within_group_pairs == 4,
                "A2 p1: pair counts wrong") ||
        !metric_close(semantic.micro_impurity, 1.0 / 4.0,
                      "A2 p1: micro impurity") ||
        !metric_close(semantic.macro_impurity, 1.0 / 4.0,
                      "A2 p1: macro impurity") ||
        !metric_close(semantic.merge_error_rate, 6.0 / 7.0,
                      "A2 p1: merge rate") ||
        !metric_close(semantic.conditional_merge_error_severity, 7.0 / 24.0,
                      "A2 p1: merge severity")) {
      return 1;
    }
    if (!expect(semantic.micro_impurity.support_count == 7 &&
                    semantic.macro_impurity.support_count == 2 &&
                    semantic.merge_error_rate.support_count == 7 &&
                    semantic.conditional_merge_error_severity.support_count ==
                        6,
                "A2 p1: semantic supports wrong") ||
        !expect(semantic.dimensions.size() == 2 &&
                    semantic.dimensions[0].dimension_index == 0 &&
                    semantic.dimensions[1].dimension_index == 1 &&
                    semantic.dimensions[0].micro_impurity.support_count == 7 &&
                    semantic.dimensions[0].macro_impurity.support_count == 2 &&
                    semantic.dimensions[1].micro_impurity.support_count == 7 &&
                    semantic.dimensions[1].macro_impurity.support_count == 2,
                "A2 p1: per-dimension envelope wrong") ||
        !metric_close(semantic.dimensions[0].micro_impurity, 2.0 / 7.0,
                      "A2 p1: dimension 0 micro") ||
        !metric_close(semantic.dimensions[0].macro_impurity, 3.0 / 8.0,
                      "A2 p1: dimension 0 macro") ||
        !metric_close(semantic.dimensions[1].micro_impurity, 3.0 / 14.0,
                      "A2 p1: dimension 1 micro") ||
        !metric_close(semantic.dimensions[1].macro_impurity, 1.0 / 8.0,
                      "A2 p1: dimension 1 macro")) {
      return 1;
    }
    if (!expect(semantic.cluster_details.has_value() &&
                    semantic.cluster_details->size() == 2 &&
                    semantic.cluster_details->at(0).observation_count == 4 &&
                    semantic.cluster_details->at(1).observation_count == 2 &&
                    semantic.cluster_details->at(0).impurity.support_count ==
                        6 &&
                    semantic.cluster_details->at(1).impurity.support_count == 1,
                "A2 p1: cluster details wrong") ||
        !metric_close(semantic.cluster_details->at(0).impurity, 1.0 / 4.0,
                      "A2 p1: cluster 0 impurity") ||
        !metric_close(semantic.cluster_details->at(1).impurity, 1.0 / 4.0,
                      "A2 p1: cluster 1 impurity")) {
      return 1;
    }

    const auto &fragmentation = unit.fragmentation;
    if (!metric_close(fragmentation.micro, 3.0 / 4.0,
                      "A2: fragmentation micro") ||
        !metric_close(fragmentation.macro, 5.0 / 6.0,
                      "A2: fragmentation macro") ||
        !expect(fragmentation.micro.support_count == 4 &&
                    fragmentation.macro.support_count == 2 &&
                    fragmentation.group_details.has_value() &&
                    fragmentation.group_details->size() == 3,
                "A2: fragmentation supports/details wrong") ||
        !metric_close(fragmentation.group_details->at(0).fragmentation,
                      2.0 / 3.0, "A2: group A fragmentation") ||
        !expect(fragmentation.group_details->at(0).observation_count == 3 &&
                    fragmentation.group_details->at(0)
                            .fragmentation.support_count == 3 &&
                    fragmentation.group_details->at(2).observation_count == 2 &&
                    fragmentation.group_details->at(2)
                            .fragmentation.support_count == 1,
                "A2: group detail supports wrong") ||
        !metric_undefined(fragmentation.group_details->at(1).fragmentation,
                          MetricUndefinedReason::NoTrueWithinGroupPairs, 0,
                          "A2: singleton group fragmentation") ||
        !metric_close(fragmentation.group_details->at(2).fragmentation, 1.0,
                      "A2: group B fragmentation")) {
      return 1;
    }
    if (!expect(close(*semantic.micro_impurity.value,
                      *semantic.merge_error_rate.value *
                          *semantic.conditional_merge_error_severity.value),
                "A2 p1: impurity identity failed")) {
      return 1;
    }

    const auto permutation = torch::tensor({5, 2, 4, 0, 3, 1}, torch::kLong);
    const auto &permuted = only_unit(
        evaluate_clustering(predicted.index_select(0, permutation),
                            truth.index_select(0, permutation), options));
    if (!expect(close(*semantic.micro_impurity.value,
                      *permuted.semantic.micro_impurity.value) &&
                    close(*semantic.macro_impurity.value,
                          *permuted.semantic.macro_impurity.value) &&
                    close(*semantic.conditional_merge_error_severity.value,
                          *permuted.semantic.conditional_merge_error_severity
                               .value) &&
                    close(*fragmentation.micro.value,
                          *permuted.fragmentation.micro.value) &&
                    close(*fragmentation.macro.value,
                          *permuted.fragmentation.macro.value),
                "A2 p1: observation permutation changed metrics")) {
      return 1;
    }

    options.power = 2;
    const auto &squared =
        only_unit(evaluate_clustering(predicted, truth, options)).semantic;
    if (!metric_close(squared.micro_impurity, 1.0 / 8.0,
                      "A2 p2: micro impurity") ||
        !metric_close(squared.macro_impurity, 1.0 / 8.0,
                      "A2 p2: macro impurity") ||
        !metric_close(squared.conditional_merge_error_severity, 7.0 / 48.0,
                      "A2 p2: merge severity") ||
        !metric_close(squared.dimensions[0].micro_impurity, 1.0 / 7.0,
                      "A2 p2: dimension 0 micro") ||
        !metric_close(squared.dimensions[0].macro_impurity, 3.0 / 16.0,
                      "A2 p2: dimension 0 macro") ||
        !metric_close(squared.dimensions[1].micro_impurity, 3.0 / 28.0,
                      "A2 p2: dimension 1 micro") ||
        !metric_close(squared.dimensions[1].macro_impurity, 1.0 / 16.0,
                      "A2 p2: dimension 1 macro") ||
        !metric_close(squared.cluster_details->at(0).impurity, 1.0 / 8.0,
                      "A2 p2: cluster 0 impurity") ||
        !metric_close(squared.cluster_details->at(1).impurity, 1.0 / 8.0,
                      "A2 p2: cluster 1 impurity")) {
      return 1;
    }
  }

  // A4 optional combined quality uses the harmonic mean of semantic cohesion
  // C=1-I_micro and pair preservation G=1-F_micro. Source records keep their
  // original pair supports beside the observation-supported derived score.
  {
    const auto truth = torch::tensor(
        {{0, 0}, {0, 0}, {0, 0}, {1, 0}, {1, 0}, {0, 2}}, torch::kLong);
    const auto predicted =
        torch::tensor({10, 10, 20, 10, 20, 10}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {2.0, 4.0};
    options.semantic_weights = {1.0, 1.0};
    options.power = 1;
    options.combined_quality = true;
    const auto result = evaluate_clustering(predicted, truth, options);
    const auto &unit = only_unit(result);
    if (!expect(result.effective_options.combined_quality &&
                    unit.combined_quality.has_value(),
                "A4 combined: requested record missing")) {
      return 1;
    }
    const auto &combined = *unit.combined_quality;
    if (!metric_close(combined.quality, 3.0 / 8.0, "A4 combined p1: quality") ||
        !metric_close(combined.semantic_micro_impurity, 1.0 / 4.0,
                      "A4 combined p1: semantic source") ||
        !metric_close(combined.fragmentation_micro, 3.0 / 4.0,
                      "A4 combined p1: fragmentation source") ||
        !expect(combined.quality.metric ==
                        ClusteringMetricId::CombinedQuality &&
                    combined.quality.support_count == 6 &&
                    combined.semantic_micro_impurity.metric ==
                        ClusteringMetricId::SemanticImpurityMicro &&
                    combined.semantic_micro_impurity.support_count == 7 &&
                    combined.fragmentation_micro.metric ==
                        ClusteringMetricId::FragmentationMicro &&
                    combined.fragmentation_micro.support_count == 4,
                "A4 combined p1: metric IDs/supports wrong")) {
      return 1;
    }

    options.power = 2;
    const auto squared_unit =
        only_unit(evaluate_clustering(predicted, truth, options));
    const auto &squared = *squared_unit.combined_quality;
    if (!metric_close(squared.quality, 7.0 / 18.0, "A4 combined p2: quality") ||
        !metric_close(squared.semantic_micro_impurity, 1.0 / 8.0,
                      "A4 combined p2: semantic source") ||
        !metric_close(squared.fragmentation_micro, 3.0 / 4.0,
                      "A4 combined p2: fragmentation source")) {
      return 1;
    }

    ClusteringEvaluationOptions scalar_options;
    scalar_options.semantic = SemanticEvaluationMode::Power;
    scalar_options.semantic_ranges = {1.0};
    scalar_options.power = 1;
    scalar_options.combined_quality = true;
    const auto zero_unit = only_unit(
        evaluate_clustering(torch::tensor({0, 1, 0, 1}, torch::kLong),
                            scalar_truth({0, 0, 1, 1}), scalar_options));
    const auto &zero = *zero_unit.combined_quality;
    if (!metric_close(zero.semantic_micro_impurity, 1.0,
                      "A4 combined zero: semantic source") ||
        !metric_close(zero.fragmentation_micro, 1.0,
                      "A4 combined zero: fragmentation source") ||
        !metric_close(zero.quality, 0.0, "A4 combined zero: quality")) {
      return 1;
    }

    const auto perfect_unit = only_unit(
        evaluate_clustering(torch::tensor({9, 9, -2, -2}, torch::kLong),
                            scalar_truth({0, 0, 1, 1}), scalar_options));
    const auto &perfect = *perfect_unit.combined_quality;
    if (!metric_close(perfect.quality, 1.0, "A4 combined perfect: quality")) {
      return 1;
    }

    const auto semantic_undefined_unit =
        only_unit(evaluate_clustering(torch::tensor({0, 1, 2}, torch::kLong),
                                      scalar_truth({0, 0, 1}), scalar_options));
    const auto &semantic_undefined = *semantic_undefined_unit.combined_quality;
    if (!metric_undefined(semantic_undefined.semantic_micro_impurity,
                          MetricUndefinedReason::NoPredictedWithinClusterPairs,
                          0, "A4 combined: semantic source undefined") ||
        !metric_close(semantic_undefined.fragmentation_micro, 1.0,
                      "A4 combined: defined fragmentation source") ||
        !metric_undefined(semantic_undefined.quality,
                          MetricUndefinedReason::DependentMetricUndefined, 3,
                          "A4 combined: semantic dependency")) {
      return 1;
    }

    scalar_options.semantic_ranges = {2.0};
    const auto fragmentation_undefined_unit =
        only_unit(evaluate_clustering(torch::zeros({3}, torch::kLong),
                                      scalar_truth({0, 1, 2}), scalar_options));
    const auto &fragmentation_undefined =
        *fragmentation_undefined_unit.combined_quality;
    if (!expect(fragmentation_undefined.semantic_micro_impurity.defined(),
                "A4 combined: semantic source unexpectedly undefined") ||
        !metric_undefined(fragmentation_undefined.fragmentation_micro,
                          MetricUndefinedReason::NoTrueWithinGroupPairs, 0,
                          "A4 combined: fragmentation source undefined") ||
        !metric_undefined(fragmentation_undefined.quality,
                          MetricUndefinedReason::DependentMetricUndefined, 3,
                          "A4 combined: fragmentation dependency")) {
      return 1;
    }
  }

  // Semantic partition quality balances the fraction of total semantic cost
  // separated by predicted boundaries with exact same-truth pair recall. It
  // must reject both the one-cluster collapse and singleton over-splitting.
  {
    const auto truth = scalar_truth({0, 0, 1, 1, 2, 2});
    const auto predicted =
        torch::tensor({10, 10, 10, 20, 20, 20}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {2.0};
    options.power = 1;
    options.semantic_partition_quality = true;

    const auto linear =
        only_unit(evaluate_clustering(predicted, truth, options));
    const auto &linear_quality = *linear.semantic_partition_quality;
    if (!metric_close(linear.semantic.partition_separation, 3.0 / 4.0,
                      "partition quality p1: separation") ||
        !metric_close(linear_quality.quality, 12.0 / 17.0,
                      "partition quality p1: quality") ||
        !metric_close(linear_quality.semantic_partition_separation, 3.0 / 4.0,
                      "partition quality p1: copied separation") ||
        !metric_close(linear_quality.pair_recall, 2.0 / 3.0,
                      "partition quality p1: copied pair recall") ||
        !expect(linear.semantic.partition_separation.support_count == 15 &&
                    linear_quality.quality.support_count == 6 &&
                    linear_quality.semantic_partition_separation.metric ==
                        ClusteringMetricId::SemanticPartitionSeparation &&
                    linear_quality.pair_recall.metric ==
                        ClusteringMetricId::PairRecall &&
                    linear_quality.pair_recall.support_count == 3,
                "partition quality p1: IDs/supports wrong")) {
      return 1;
    }

    options.power = 2;
    const auto squared =
        only_unit(evaluate_clustering(predicted, truth, options));
    if (!metric_close(squared.semantic.partition_separation, 5.0 / 6.0,
                      "partition quality p2: separation") ||
        !metric_close(squared.semantic_partition_quality->quality, 20.0 / 27.0,
                      "partition quality p2: quality")) {
      return 1;
    }

    ClusteringEvaluationOptions endpoint_options;
    endpoint_options.semantic = SemanticEvaluationMode::Power;
    endpoint_options.semantic_ranges = {1.0};
    endpoint_options.power = 1;
    endpoint_options.semantic_partition_quality = true;
    const auto endpoint_truth = scalar_truth({0, 0, 1, 1});

    const auto perfect = only_unit(
        evaluate_clustering(torch::tensor({9, 9, -2, -2}, torch::kLong),
                            endpoint_truth, endpoint_options));
    if (!metric_close(perfect.semantic.partition_separation, 1.0,
                      "partition quality perfect: separation") ||
        !metric_close(perfect.semantic_partition_quality->quality, 1.0,
                      "partition quality perfect: quality")) {
      return 1;
    }

    const auto collapsed = only_unit(evaluate_clustering(
        torch::zeros({4}, torch::kLong), endpoint_truth, endpoint_options));
    if (!metric_close(collapsed.semantic.partition_separation, 0.0,
                      "partition quality collapse: separation") ||
        !metric_close(collapsed.semantic_partition_quality->pair_recall, 1.0,
                      "partition quality collapse: pair recall") ||
        !metric_close(collapsed.semantic_partition_quality->quality, 0.0,
                      "partition quality collapse: quality")) {
      return 1;
    }

    const auto singletons =
        only_unit(evaluate_clustering(torch::tensor({0, 1, 2, 3}, torch::kLong),
                                      endpoint_truth, endpoint_options));
    if (!metric_close(singletons.semantic.partition_separation, 1.0,
                      "partition quality singletons: separation") ||
        !metric_close(singletons.semantic_partition_quality->pair_recall, 0.0,
                      "partition quality singletons: pair recall") ||
        !metric_close(singletons.semantic_partition_quality->quality, 0.0,
                      "partition quality singletons: quality") ||
        !metric_undefined(singletons.semantic.micro_impurity,
                          MetricUndefinedReason::NoPredictedWithinClusterPairs,
                          0, "partition quality singletons: legacy impurity")) {
      return 1;
    }

    const auto no_variation = only_unit(
        evaluate_clustering(torch::tensor({0, 0, 1, 1}, torch::kLong),
                            scalar_truth({0, 0, 0, 0}), endpoint_options));
    if (!metric_undefined(no_variation.semantic.partition_separation,
                          MetricUndefinedReason::NoSemanticVariation, 6,
                          "partition quality: no semantic variation") ||
        !metric_undefined(no_variation.semantic_partition_quality->quality,
                          MetricUndefinedReason::DependentMetricUndefined, 4,
                          "partition quality: no-variation dependency")) {
      return 1;
    }

    ClusteringEvaluationOptions zero_weight_variation_options =
        endpoint_options;
    zero_weight_variation_options.semantic_ranges = {1.0, 1.0};
    zero_weight_variation_options.semantic_weights = {1.0, 0.0};
    const auto zero_weight_variation = only_unit(evaluate_clustering(
        torch::tensor({0, 0, 1, 1}, torch::kLong),
        torch::tensor({{0, 0}, {0, 0}, {0, 1}, {0, 1}}, torch::kLong),
        zero_weight_variation_options));
    if (!metric_undefined(
            zero_weight_variation.semantic.partition_separation,
            MetricUndefinedReason::NoSemanticVariation, 6,
            "partition quality: zero-weight-only semantic variation") ||
        !metric_undefined(
            zero_weight_variation.semantic_partition_quality->quality,
            MetricUndefinedReason::DependentMetricUndefined, 4,
            "partition quality: zero-weight-only dependency")) {
      return 1;
    }
  }

  // Zero-weight dimensions stay visible in per-dimension metrics but do not
  // contribute to aggregate semantic impurity.
  {
    const auto truth = torch::tensor({{0, 0}, {0, 2}, {1, 0}}, torch::kLong);
    const auto predicted = torch::zeros({3}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {2.0, 2.0};
    options.semantic_weights = {1.0, 0.0};
    options.power = 2;
    const auto &unit =
        only_unit(evaluate_clustering(predicted, truth, options));
    if (!metric_close(unit.semantic.micro_impurity, 1.0 / 6.0,
                      "zero weight: micro impurity") ||
        !metric_close(unit.semantic.macro_impurity, 1.0 / 6.0,
                      "zero weight: macro impurity") ||
        !metric_close(unit.semantic.merge_error_rate, 1.0,
                      "zero weight: merge rate") ||
        !metric_close(unit.semantic.conditional_merge_error_severity, 1.0 / 6.0,
                      "zero weight: severity") ||
        !metric_close(unit.semantic.dimensions[0].micro_impurity, 1.0 / 6.0,
                      "zero weight: dimension 0") ||
        !metric_close(unit.semantic.dimensions[1].micro_impurity, 2.0 / 3.0,
                      "zero weight: dimension 1") ||
        !metric_undefined(unit.fragmentation.micro,
                          MetricUndefinedReason::NoTrueWithinGroupPairs, 0,
                          "zero weight: fragmentation micro") ||
        !metric_undefined(unit.fragmentation.macro,
                          MetricUndefinedReason::NoEligibleTruthGroups, 0,
                          "zero weight: fragmentation macro")) {
      return 1;
    }
  }

  // Exact-only mode keeps semantic records discoverable, while fragmentation
  // remains available without semantic ranges.
  {
    const auto truth = torch::tensor({{0, 0}, {0, 0}, {1, 0}}, torch::kLong);
    const auto predicted = torch::tensor({0, 0, 1}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    const auto result = evaluate_clustering(predicted, truth, options);
    const auto &unit = only_unit(result);
    if (!expect(result.effective_options.semantic ==
                        SemanticEvaluationMode::Off &&
                    result.effective_options.semantic_ranges.empty() &&
                    result.effective_options.semantic_weights.empty(),
                "semantic off: effective options wrong") ||
        !metric_undefined(unit.semantic.micro_impurity,
                          MetricUndefinedReason::SemanticDisabled, 1,
                          "semantic off: micro") ||
        !metric_undefined(unit.semantic.macro_impurity,
                          MetricUndefinedReason::SemanticDisabled, 1,
                          "semantic off: macro") ||
        !metric_undefined(unit.semantic.merge_error_rate,
                          MetricUndefinedReason::SemanticDisabled, 1,
                          "semantic off: merge rate") ||
        !metric_undefined(unit.semantic.conditional_merge_error_severity,
                          MetricUndefinedReason::SemanticDisabled, 0,
                          "semantic off: severity") ||
        !metric_undefined(unit.semantic.partition_separation,
                          MetricUndefinedReason::SemanticDisabled, 3,
                          "semantic off: partition separation") ||
        !expect(unit.semantic.dimensions.size() == 2 &&
                    unit.semantic.cluster_details.has_value() &&
                    unit.semantic.cluster_details->size() == 2,
                "semantic off: details missing") ||
        !metric_close(unit.fragmentation.micro, 0.0,
                      "semantic off: fragmentation micro") ||
        !metric_close(unit.fragmentation.macro, 0.0,
                      "semantic off: fragmentation macro")) {
      return 1;
    }
  }

  // No predicted pairs and no merge errors are distinct undefined conditions;
  // fragmentation can still be a defined maximum.
  {
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {1.0};
    const auto split =
        only_unit(evaluate_clustering(torch::tensor({0, 1, 2}, torch::kLong),
                                      scalar_truth({0, 0, 1}), options));
    if (!metric_undefined(split.semantic.micro_impurity,
                          MetricUndefinedReason::NoPredictedWithinClusterPairs,
                          0, "A2 split: micro") ||
        !metric_undefined(split.semantic.macro_impurity,
                          MetricUndefinedReason::NoEligiblePredictedClusters, 0,
                          "A2 split: macro") ||
        !metric_undefined(split.semantic.merge_error_rate,
                          MetricUndefinedReason::NoPredictedWithinClusterPairs,
                          0, "A2 split: merge rate") ||
        !metric_undefined(split.semantic.conditional_merge_error_severity,
                          MetricUndefinedReason::NoMergeErrorPairs, 0,
                          "A2 split: severity") ||
        !metric_close(split.fragmentation.micro, 1.0,
                      "A2 split: fragmentation micro") ||
        !metric_close(split.fragmentation.macro, 1.0,
                      "A2 split: fragmentation macro")) {
      return 1;
    }

    const auto perfect = only_unit(
        evaluate_clustering(torch::tensor({9, 9, -2, -2}, torch::kLong),
                            scalar_truth({0, 0, 1, 1}), options));
    if (!expect(perfect.semantic.dimensions.size() == 1 &&
                    perfect.semantic.cluster_details.has_value() &&
                    perfect.fragmentation.group_details.has_value(),
                "A2 perfect: detail envelope wrong") ||
        !metric_close(perfect.semantic.micro_impurity, 0.0,
                      "A2 perfect: micro impurity") ||
        !metric_close(perfect.semantic.macro_impurity, 0.0,
                      "A2 perfect: macro impurity") ||
        !metric_close(perfect.semantic.merge_error_rate, 0.0,
                      "A2 perfect: merge rate") ||
        !metric_undefined(perfect.semantic.conditional_merge_error_severity,
                          MetricUndefinedReason::NoMergeErrorPairs, 0,
                          "A2 perfect: severity") ||
        !metric_close(perfect.fragmentation.micro, 0.0,
                      "A2 perfect: fragmentation micro") ||
        !metric_close(perfect.fragmentation.macro, 0.0,
                      "A2 perfect: fragmentation macro") ||
        !expect(perfect.semantic.micro_impurity.support_count == 2 &&
                    perfect.semantic.macro_impurity.support_count == 2 &&
                    perfect.fragmentation.micro.support_count == 2 &&
                    perfect.fragmentation.macro.support_count == 2,
                "A2 perfect: supports wrong")) {
      return 1;
    }

    const auto one_group = only_unit(evaluate_clustering(
        torch::zeros({4}, torch::kLong), scalar_truth({5, 5, 5, 5}), options));
    if (!metric_close(one_group.semantic.micro_impurity, 0.0,
                      "A2 one-group: micro impurity") ||
        !metric_close(one_group.semantic.macro_impurity, 0.0,
                      "A2 one-group: macro impurity") ||
        !metric_close(one_group.semantic.merge_error_rate, 0.0,
                      "A2 one-group: merge rate") ||
        !metric_undefined(one_group.semantic.conditional_merge_error_severity,
                          MetricUndefinedReason::NoMergeErrorPairs, 0,
                          "A2 one-group: severity") ||
        !metric_close(one_group.fragmentation.micro, 0.0,
                      "A2 one-group: fragmentation micro") ||
        !metric_close(one_group.fragmentation.macro, 0.0,
                      "A2 one-group: fragmentation macro") ||
        !expect(one_group.semantic.micro_impurity.support_count == 6 &&
                    one_group.semantic.macro_impurity.support_count == 1 &&
                    one_group.fragmentation.micro.support_count == 6 &&
                    one_group.fragmentation.macro.support_count == 1,
                "A2 one-group: supports wrong")) {
      return 1;
    }
  }

  // A3 G<K fixture: exact overlap leaves the mixed cluster unmatched, while
  // semantic cost prefers paying the dummy penalty for a smaller pure cluster.
  // This exercises distinct mappings, fixed dummy cost, per-dimension errors,
  // marginal supports, Full details, both powers, and canonical identities.
  {
    const auto truth = torch::tensor({{0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {1, 2},
                                      {1, 2},
                                      {1, 2},
                                      {1, 2},
                                      {1, 2}},
                                     torch::kLong);
    const auto predicted =
        torch::tensor({-5, -5, -5, 30, 30, 10, 10, 10, 30, 30}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {4.0, 4.0};
    options.semantic_weights = {3.0, 1.0};
    options.power = 1;
    options.alignment = AlignmentEvaluationMode::Both;

    const auto result = evaluate_clustering(predicted, truth, options);
    const auto &unit = only_unit(result);
    if (!expect(result.effective_options.alignment ==
                        AlignmentEvaluationMode::Both &&
                    unit.alignment_identities.has_value() &&
                    unit.exact_alignment.has_value() &&
                    unit.semantic_alignment.has_value(),
                "A3 G<K: requested results/identities missing") ||
        !expect(torch::equal(unit.alignment_identities->predicted_ids,
                             torch::tensor({-5, 10, 30}, torch::kLong)) &&
                    torch::equal(unit.alignment_identities->truth_vectors,
                                 torch::tensor({{0, 0}, {1, 2}}, torch::kLong)),
                "A3 G<K: canonical identities wrong")) {
      return 1;
    }

    const auto &exact = *unit.exact_alignment;
    if (!expect(exact.mapping.predicted_to_truth_group ==
                        std::vector<std::int64_t>({0, 1, -1}) &&
                    exact.mapping.truth_to_predicted_cluster ==
                        std::vector<std::int64_t>({0, 1}) &&
                    exact.aligned_column_to_predicted_cluster ==
                        std::vector<std::int64_t>({0, 1, 2}),
                "A3 G<K exact: mapping/permutation wrong") ||
        !expect(exact.matched_overlap_observation_count == 6 &&
                    exact.nonmatched_observation_count == 4 &&
                    exact.mapping.assigned_predicted_observation_count == 6 &&
                    exact.mapping.unmatched_predicted_observation_count == 4 &&
                    exact.mapping.assigned_truth_observation_count == 10 &&
                    exact.mapping.unmatched_truth_observation_count == 0 &&
                    exact.mapping.unmatched_predicted_clusters.size() == 1 &&
                    exact.mapping.unmatched_predicted_clusters[0].index == 2 &&
                    exact.mapping.unmatched_predicted_clusters[0]
                            .observation_count == 4,
                "A3 G<K exact: supports wrong") ||
        !metric_close(exact.matched_accuracy, 3.0 / 5.0,
                      "A3 G<K exact: accuracy") ||
        !expect(exact.matched_accuracy.support_count == 10 &&
                    exact.truth_group_details.has_value() &&
                    exact.truth_group_details->size() == 2,
                "A3 G<K exact: detail envelope wrong")) {
      return 1;
    }
    for (const auto &group : *exact.truth_group_details) {
      if (!expect(group.truth_observation_count == 5 &&
                      group.predicted_observation_count == 3 &&
                      group.overlap_observation_count == 3,
                  "A3 G<K exact: group counts wrong") ||
          !metric_close(group.precision, 1.0,
                        "A3 G<K exact: group precision") ||
          !metric_close(group.recall, 3.0 / 5.0,
                        "A3 G<K exact: group recall") ||
          !metric_close(group.f1, 3.0 / 4.0, "A3 G<K exact: group F1") ||
          !metric_close(group.jaccard, 3.0 / 5.0,
                        "A3 G<K exact: group Jaccard") ||
          !expect(group.precision.support_count == 3 &&
                      group.recall.support_count == 5 &&
                      group.f1.support_count == 8 &&
                      group.jaccard.support_count == 5,
                  "A3 G<K exact: group supports wrong")) {
        return 1;
      }
    }

    const auto &semantic = *unit.semantic_alignment;
    if (!expect(semantic.mapping.predicted_to_truth_group ==
                        std::vector<std::int64_t>({0, -1, 1}) &&
                    semantic.mapping.truth_to_predicted_cluster ==
                        std::vector<std::int64_t>({0, 2}),
                "A3 G<K semantic: mapping wrong") ||
        !expect(semantic.mapping.assigned_predicted_observation_count == 7 &&
                    semantic.mapping.unmatched_predicted_observation_count ==
                        3 &&
                    semantic.mapping.assigned_truth_observation_count == 10 &&
                    semantic.mapping.unmatched_truth_observation_count == 0 &&
                    semantic.exact_overlap_observation_count == 5 &&
                    semantic.nonoverlap_observation_count == 5 &&
                    close(semantic.unmatched_predicted_penalty, 1.0),
                "A3 G<K semantic: supports/penalty wrong") ||
        !metric_close(semantic.normalized_cost, 29.0 / 80.0,
                      "A3 G<K semantic p1: normalized cost") ||
        !expect(semantic.normalized_cost.support_count == 10 &&
                    semantic.dimensions.size() == 2 &&
                    semantic.dimensions[0].normalized_error.support_count ==
                        10 &&
                    semantic.dimensions[1].normalized_error.support_count == 10,
                "A3 G<K semantic p1: metric supports wrong") ||
        !metric_close(semantic.dimensions[0].normalized_error, 7.0 / 20.0,
                      "A3 G<K semantic p1: dimension 0") ||
        !metric_close(semantic.dimensions[1].normalized_error, 2.0 / 5.0,
                      "A3 G<K semantic p1: dimension 1") ||
        !expect(semantic.error_masses.has_value() &&
                    semantic.error_masses->size() == 4,
                "A3 G<K semantic p1: masses missing")) {
      return 1;
    }
    std::uint64_t mass_support = 0;
    double weighted_mass_cost = 0.0;
    for (const auto &mass : *semantic.error_masses) {
      mass_support += mass.observation_count;
      weighted_mass_cost +=
          mass.normalized_cost * static_cast<double>(mass.observation_count);
    }
    if (!expect(
            mass_support == 10 &&
                close(weighted_mass_cost / 10.0, 29.0 / 80.0) &&
                semantic.error_masses->at(1).source_truth_group_index == 0 &&
                semantic.error_masses->at(1).predicted_cluster_index == 2 &&
                semantic.error_masses->at(1).assigned_truth_group_index == 1 &&
                close(semantic.error_masses->at(1).normalized_cost,
                      5.0 / 16.0) &&
                semantic.error_masses->at(2).assigned_truth_group_index == -1 &&
                close(semantic.error_masses->at(2).normalized_cost, 1.0),
            "A3 G<K semantic p1: mass records wrong")) {
      return 1;
    }

    const auto permutation =
        torch::tensor({9, 3, 7, 0, 6, 2, 8, 5, 1, 4}, torch::kLong);
    const auto &permuted = only_unit(
        evaluate_clustering(predicted.index_select(0, permutation),
                            truth.index_select(0, permutation), options));
    if (!expect(
            permuted.exact_alignment->mapping.predicted_to_truth_group ==
                    exact.mapping.predicted_to_truth_group &&
                permuted.semantic_alignment->mapping.predicted_to_truth_group ==
                    semantic.mapping.predicted_to_truth_group &&
                close(*permuted.semantic_alignment->normalized_cost.value,
                      *semantic.normalized_cost.value),
            "A3 G<K: observation permutation changed alignment")) {
      return 1;
    }

    options.power = 2;
    const auto squared_unit =
        only_unit(evaluate_clustering(predicted, truth, options));
    const auto &squared = *squared_unit.semantic_alignment;
    if (!expect(squared.mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({0, -1, 1}),
                "A3 G<K semantic p2: mapping wrong") ||
        !metric_close(squared.normalized_cost, 103.0 / 320.0,
                      "A3 G<K semantic p2: normalized cost") ||
        !metric_close(squared.dimensions[0].normalized_error, 5.0 / 16.0,
                      "A3 G<K semantic p2: dimension 0") ||
        !metric_close(squared.dimensions[1].normalized_error, 7.0 / 20.0,
                      "A3 G<K semantic p2: dimension 1")) {
      return 1;
    }

    options.detail = ClusteringEvaluationDetail::Global;
    options.power = 1;
    const auto &global =
        only_unit(evaluate_clustering(predicted, truth, options));
    if (!expect(global.alignment_identities.has_value() &&
                    !global.partition_detail.has_value() &&
                    !global.exact_alignment->truth_group_details.has_value() &&
                    !global.semantic_alignment->error_masses.has_value(),
                "A3 G<K global: detail policy wrong")) {
      return 1;
    }
  }

  // A3 G>K fixture: exact overlap chooses the plurality truth group, while
  // semantic alignment chooses the weighted semantic medoid. Unmatched truth
  // supports and undefined per-group precision/F1 remain explicit.
  {
    const auto truth = scalar_truth({0, 0, 0, 0, 4, 4, 4, 10, 10, 10});
    const auto predicted = torch::full({10}, 42, torch::kLong);
    ClusteringEvaluationOptions options;
    options.detail = ClusteringEvaluationDetail::Full;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {10.0};
    options.power = 1;
    options.alignment = AlignmentEvaluationMode::Both;
    const auto &unit =
        only_unit(evaluate_clustering(predicted, truth, options));

    const auto &exact = *unit.exact_alignment;
    if (!expect(exact.mapping.predicted_to_truth_group ==
                        std::vector<std::int64_t>({0}) &&
                    exact.mapping.truth_to_predicted_cluster ==
                        std::vector<std::int64_t>({0, -1, -1}) &&
                    exact.mapping.assigned_predicted_observation_count == 10 &&
                    exact.mapping.unmatched_predicted_observation_count == 0 &&
                    exact.mapping.assigned_truth_observation_count == 4 &&
                    exact.mapping.unmatched_truth_observation_count == 6 &&
                    exact.mapping.unmatched_truth_groups.size() == 2,
                "A3 G>K exact: mapping/supports wrong") ||
        !metric_close(exact.matched_accuracy, 2.0 / 5.0,
                      "A3 G>K exact: accuracy") ||
        !metric_undefined(exact.truth_group_details->at(1).precision,
                          MetricUndefinedReason::NoMatchedPredictedCluster, 0,
                          "A3 G>K exact: unmatched precision") ||
        !metric_close(exact.truth_group_details->at(1).recall, 0.0,
                      "A3 G>K exact: unmatched recall") ||
        !metric_undefined(exact.truth_group_details->at(1).f1,
                          MetricUndefinedReason::DependentMetricUndefined, 3,
                          "A3 G>K exact: unmatched F1") ||
        !metric_close(exact.truth_group_details->at(1).jaccard, 0.0,
                      "A3 G>K exact: unmatched Jaccard")) {
      return 1;
    }

    const auto &semantic = *unit.semantic_alignment;
    if (!expect(
            semantic.mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({1}) &&
                semantic.mapping.truth_to_predicted_cluster ==
                    std::vector<std::int64_t>({-1, 0, -1}) &&
                semantic.mapping.assigned_predicted_observation_count == 10 &&
                semantic.mapping.unmatched_predicted_observation_count == 0 &&
                semantic.mapping.assigned_truth_observation_count == 3 &&
                semantic.mapping.unmatched_truth_observation_count == 7 &&
                semantic.exact_overlap_observation_count == 3 &&
                semantic.nonoverlap_observation_count == 7,
            "A3 G>K semantic p1: mapping/supports wrong") ||
        !metric_close(semantic.normalized_cost, 17.0 / 50.0,
                      "A3 G>K semantic p1: cost") ||
        !metric_close(semantic.dimensions[0].normalized_error, 17.0 / 50.0,
                      "A3 G>K semantic p1: dimension") ||
        !expect(semantic.error_masses->size() == 3 &&
                    close(semantic.error_masses->at(0).normalized_cost, 0.4) &&
                    close(semantic.error_masses->at(1).normalized_cost, 0.0) &&
                    close(semantic.error_masses->at(2).normalized_cost, 0.6),
                "A3 G>K semantic p1: masses wrong")) {
      return 1;
    }

    options.power = 2;
    const auto squared_unit =
        only_unit(evaluate_clustering(predicted, truth, options));
    const auto &squared = *squared_unit.semantic_alignment;
    if (!expect(squared.mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({1}),
                "A3 G>K semantic p2: mapping wrong") ||
        !metric_close(squared.normalized_cost, 43.0 / 250.0,
                      "A3 G>K semantic p2: cost") ||
        !metric_close(squared.dimensions[0].normalized_error, 43.0 / 250.0,
                      "A3 G>K semantic p2: dimension")) {
      return 1;
    }
  }

  // Equal-primary optima use the locked predicted-major dense tie rule in
  // square and both rectangular directions.
  {
    ClusteringEvaluationOptions options;
    options.alignment = AlignmentEvaluationMode::Exact;
    const auto &square = only_unit(
        evaluate_clustering(torch::tensor({10, 20, 10, 20}, torch::kLong),
                            scalar_truth({0, 0, 1, 1}), options));
    const auto &more_predicted = only_unit(evaluate_clustering(
        torch::tensor({10, 20}, torch::kLong), scalar_truth({0, 0}), options));
    const auto &more_truth = only_unit(evaluate_clustering(
        torch::tensor({10, 10}, torch::kLong), scalar_truth({0, 1}), options));
    // Raw ascending Hungarian traversal chooses {1,0} for this equal optimum;
    // the equality-graph refinement must flip it to the contractual {0,1}.
    const auto &requires_refinement = only_unit(evaluate_clustering(
        torch::tensor({10, 20, 10, 10, 20, 20}, torch::kLong),
        scalar_truth({0, 0, 1, 1, 1, 1}), options));
    if (!expect(
            square.exact_alignment->mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({0, 1}) &&
                more_predicted.exact_alignment->mapping
                        .predicted_to_truth_group ==
                    std::vector<std::int64_t>({0, -1}) &&
                more_truth.exact_alignment->mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({0}) &&
                more_truth.exact_alignment->mapping
                        .truth_to_predicted_cluster ==
                    std::vector<std::int64_t>({0, -1}) &&
                requires_refinement.exact_alignment->mapping
                        .predicted_to_truth_group ==
                    std::vector<std::int64_t>({0, 1}),
            "A3 ties: dense lexicographic rule failed") ||
        !metric_close(requires_refinement.exact_alignment->matched_accuracy,
                      0.5, "A3 ties: refinement accuracy") ||
        !expect(!square.semantic_alignment.has_value() &&
                    square.alignment_identities.has_value(),
                "A3 exact-only: result engagement wrong")) {
      return 1;
    }
  }

  // Zero-weight dimensions remain visible in semantic alignment diagnostics
  // but do not affect the aggregate assignment objective or cost.
  {
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {1.0, 1.0};
    options.semantic_weights = {1.0, 0.0};
    options.power = 1;
    options.alignment = AlignmentEvaluationMode::Semantic;
    const auto alignment_unit = only_unit(evaluate_clustering(
        torch::zeros({2}, torch::kLong),
        torch::tensor({{0, 0}, {0, 1}}, torch::kLong), options));
    const auto &alignment = *alignment_unit.semantic_alignment;
    if (!expect(alignment.mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({0}),
                "A3 zero weight: mapping wrong") ||
        !metric_close(alignment.normalized_cost, 0.0,
                      "A3 zero weight: aggregate cost") ||
        !metric_close(alignment.dimensions[0].normalized_error, 0.0,
                      "A3 zero weight: dimension 0") ||
        !metric_close(alignment.dimensions[1].normalized_error, 0.5,
                      "A3 zero weight: dimension 1")) {
      return 1;
    }
  }

  // Batched D=4 alignment evaluates canonical mappings independently per unit.
  {
    const auto truth = torch::stack({
        torch::tensor({{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1}},
                      torch::kLong),
        torch::tensor({{2, 2, 2, 2}, {2, 2, 2, 2}, {3, 3, 3, 3}, {3, 3, 3, 3}},
                      torch::kLong),
    });
    const auto predicted = torch::stack({
        torch::tensor({9, 9, -2, -2}, torch::kLong),
        torch::tensor({3, 3, 4, 4}, torch::kLong),
    });
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {1.0, 1.0, 1.0, 1.0};
    options.alignment = AlignmentEvaluationMode::Both;
    const auto result = evaluate_clustering(predicted, truth, options);
    if (!expect(
            result.batched && result.units.size() == 2 &&
                result.units[0].alignment_identities.has_value() &&
                result.units[1].alignment_identities.has_value() &&
                result.units[0]
                        .exact_alignment->mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({1, 0}) &&
                result.units[0]
                        .exact_alignment->aligned_column_to_predicted_cluster ==
                    std::vector<std::int64_t>({1, 0}) &&
                result.units[1]
                        .exact_alignment->mapping.predicted_to_truth_group ==
                    std::vector<std::int64_t>({0, 1}),
            "A3 batch D4: mappings/identities wrong") ||
        !metric_close(result.units[0].semantic_alignment->normalized_cost, 0.0,
                      "A3 batch D4: unit 0 cost") ||
        !metric_close(result.units[1].semantic_alignment->normalized_cost, 0.0,
                      "A3 batch D4: unit 1 cost")) {
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

  // Semantic range validation is per unit, and D=4 batched evaluation retains
  // one result per unit without materializing full details in Global mode.
  {
    const auto truth0 = torch::tensor(
        {{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}, torch::kLong);
    const auto truth1 = torch::tensor(
        {{5, 1, 2, 3}, {5, 1, 2, 3}, {5, 1, 2, 3}, {6, 1, 2, 3}}, torch::kLong);
    const auto predicted = torch::stack({
        torch::tensor({9, 9, -2, -2}, torch::kLong),
        torch::tensor({0, 1, 2, 2}, torch::kLong),
    });
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {1.0, 1.0, 1.0, 1.0};
    const auto result =
        evaluate_clustering(predicted, torch::stack({truth0, truth1}), options);
    if (!expect(result.effective_options.semantic_weights ==
                        std::vector<double>({1.0, 1.0, 1.0, 1.0}) &&
                    result.units[0].semantic.dimensions.size() == 4 &&
                    result.units[1].semantic.dimensions.size() == 4 &&
                    !result.units[0].semantic.cluster_details.has_value() &&
                    !result.units[1].fragmentation.group_details.has_value(),
                "A2 batched: effective options/global detail wrong") ||
        !metric_close(result.units[0].semantic.micro_impurity, 0.0,
                      "A2 batched: unit 0 impurity") ||
        !metric_close(result.units[0].fragmentation.micro, 0.0,
                      "A2 batched: unit 0 fragmentation") ||
        !metric_close(result.units[1].semantic.micro_impurity, 1.0 / 4.0,
                      "A2 batched: unit 1 impurity") ||
        !metric_close(result.units[1].semantic.merge_error_rate, 1.0,
                      "A2 batched: unit 1 merge rate") ||
        !metric_close(result.units[1].fragmentation.micro, 1.0,
                      "A2 batched: unit 1 fragmentation")) {
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

  // Power semantics preserve small differences around large integral values and
  // handle signed floating coordinates without converting truth to scalar IDs.
  {
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {1.0};
    options.power = 2;
    options.alignment = AlignmentEvaluationMode::Semantic;
    const std::int64_t large0 = 9'007'199'254'740'992LL;
    const std::int64_t large1 = 9'007'199'254'740'993LL;
    const auto large_integral = only_unit(evaluate_clustering(
        torch::zeros({2}, torch::kLong),
        torch::tensor({large0, large1}, torch::kLong).reshape({2, 1}),
        options));
    if (!metric_close(large_integral.semantic.micro_impurity, 1.0,
                      "A2 dtype: adjacent large int64 values") ||
        !metric_close(large_integral.semantic_alignment->normalized_cost, 0.5,
                      "A3 dtype: adjacent large int64 alignment")) {
      return 1;
    }

    auto wide_truth = torch::empty(
        {2, 1}, torch::TensorOptions().dtype(c10::ScalarType::UInt64));
    wide_truth.data_ptr<std::uint64_t>()[0] =
        std::numeric_limits<std::uint64_t>::max() - 1;
    wide_truth.data_ptr<std::uint64_t>()[1] =
        std::numeric_limits<std::uint64_t>::max();
    const auto wide_integral = only_unit(evaluate_clustering(
        torch::zeros({2}, torch::kLong), wide_truth, options));
    if (!metric_close(wide_integral.semantic.micro_impurity, 1.0,
                      "A2 dtype: adjacent uint64 values") ||
        !metric_close(wide_integral.semantic_alignment->normalized_cost, 0.5,
                      "A3 dtype: adjacent uint64 alignment")) {
      return 1;
    }

    options.semantic_ranges = {2.0};
    options.power = 1;
    const auto floating = only_unit(evaluate_clustering(
        torch::zeros({2}, torch::kLong),
        torch::tensor({-1.5, 0.5}, torch::kFloat64).reshape({2, 1}), options));
    if (!metric_close(floating.semantic.micro_impurity, 1.0,
                      "A2 dtype: signed floating values") ||
        !metric_close(floating.semantic_alignment->normalized_cost, 0.5,
                      "A3 dtype: signed floating alignment")) {
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

    ClusteringEvaluationOptions missing_ranges;
    missing_ranges.semantic = SemanticEvaluationMode::Power;
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, missing_ranges));
                }),
                "validation: power semantics without ranges accepted")) {
      return 1;
    }

    ClusteringEvaluationOptions wrong_range_count;
    wrong_range_count.semantic_ranges = {1.0, 1.0};
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, wrong_range_count));
                }),
                "validation: semantic range count mismatch accepted")) {
      return 1;
    }

    for (const auto invalid_range :
         {0.0, -1.0, std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
      ClusteringEvaluationOptions invalid;
      invalid.semantic = SemanticEvaluationMode::Power;
      invalid.semantic_ranges = {invalid_range};
      if (!expect(throws_invalid_argument([&] {
                    static_cast<void>(evaluate_clustering(
                        valid_predicted, valid_truth, invalid));
                  }),
                  "validation: invalid semantic range accepted")) {
        return 1;
      }
    }

    ClusteringEvaluationOptions wrong_weight_count;
    wrong_weight_count.semantic = SemanticEvaluationMode::Power;
    wrong_weight_count.semantic_ranges = {1.0};
    wrong_weight_count.semantic_weights = {1.0, 1.0};
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, wrong_weight_count));
                }),
                "validation: semantic weight count mismatch accepted")) {
      return 1;
    }

    for (const auto invalid_weight :
         {-1.0, std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
      ClusteringEvaluationOptions invalid;
      invalid.semantic = SemanticEvaluationMode::Power;
      invalid.semantic_ranges = {1.0};
      invalid.semantic_weights = {invalid_weight};
      if (!expect(throws_invalid_argument([&] {
                    static_cast<void>(evaluate_clustering(
                        valid_predicted, valid_truth, invalid));
                  }),
                  "validation: invalid semantic weight accepted")) {
        return 1;
      }
    }

    ClusteringEvaluationOptions zero_weights;
    zero_weights.semantic = SemanticEvaluationMode::Power;
    zero_weights.semantic_ranges = {1.0};
    zero_weights.semantic_weights = {0.0};
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, zero_weights));
                }),
                "validation: all-zero semantic weights accepted")) {
      return 1;
    }

    for (const auto invalid_power : {0, 3}) {
      ClusteringEvaluationOptions invalid;
      invalid.semantic = SemanticEvaluationMode::Power;
      invalid.semantic_ranges = {1.0};
      invalid.power = invalid_power;
      if (!expect(throws_invalid_argument([&] {
                    static_cast<void>(evaluate_clustering(
                        valid_predicted, valid_truth, invalid));
                  }),
                  "validation: invalid semantic power accepted")) {
        return 1;
      }
    }

    ClusteringEvaluationOptions out_of_range;
    out_of_range.semantic = SemanticEvaluationMode::Power;
    out_of_range.semantic_ranges = {0.5};
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, out_of_range));
                }),
                "validation: out-of-range semantic coordinate accepted")) {
      return 1;
    }

    ClusteringEvaluationOptions invalid_semantic_mode;
    invalid_semantic_mode.semantic = static_cast<SemanticEvaluationMode>(99);
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, invalid_semantic_mode));
                }),
                "validation: unknown semantic mode accepted")) {
      return 1;
    }

    for (const auto alignment :
         {AlignmentEvaluationMode::Semantic, AlignmentEvaluationMode::Both}) {
      ClusteringEvaluationOptions semantic_alignment_without_power;
      semantic_alignment_without_power.alignment = alignment;
      if (!expect(throws_invalid_argument([&] {
                    static_cast<void>(
                        evaluate_clustering(valid_predicted, valid_truth,
                                            semantic_alignment_without_power));
                  }),
                  "validation: semantic alignment without power accepted")) {
        return 1;
      }
    }

    ClusteringEvaluationOptions combined_without_power;
    combined_without_power.combined_quality = true;
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, combined_without_power));
                }),
                "validation: combined quality without power accepted")) {
      return 1;
    }

    ClusteringEvaluationOptions partition_quality_without_power;
    partition_quality_without_power.semantic_partition_quality = true;
    if (!expect(
            throws_invalid_argument([&] {
              static_cast<void>(
                  evaluate_clustering(valid_predicted, valid_truth,
                                      partition_quality_without_power));
            }),
            "validation: semantic partition quality without power accepted")) {
      return 1;
    }

    ClusteringEvaluationOptions invalid_alignment_mode;
    invalid_alignment_mode.alignment = static_cast<AlignmentEvaluationMode>(99);
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, invalid_alignment_mode));
                }),
                "validation: unknown alignment mode accepted")) {
      return 1;
    }

    ClusteringEvaluationOptions invalid_detail;
    invalid_detail.detail = static_cast<ClusteringEvaluationDetail>(99);
    if (!expect(throws_invalid_argument([&] {
                  static_cast<void>(evaluate_clustering(
                      valid_predicted, valid_truth, invalid_detail));
                }),
                "validation: unknown detail mode accepted")) {
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

  // A2 p=1 stress: one large predicted cluster must use sorting/prefix sums,
  // not explicit observation-pair enumeration.
  {
    constexpr std::int64_t observations = 20'000;
    const auto truth =
        torch::arange(observations, torch::kLong).reshape({observations, 1});
    const auto predicted = torch::zeros({observations}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {static_cast<double>(observations - 1)};
    options.power = 1;
    const auto &unit =
        only_unit(evaluate_clustering(predicted, truth, options));
    const auto expected = static_cast<double>(observations + 1) /
                          (3.0 * static_cast<double>(observations - 1));
    if (!metric_close(unit.semantic.micro_impurity, expected,
                      "A2 p1 stress: micro impurity", 1.0e-12) ||
        !metric_close(unit.semantic.macro_impurity, expected,
                      "A2 p1 stress: macro impurity", 1.0e-12) ||
        !metric_close(unit.semantic.merge_error_rate, 1.0,
                      "A2 p1 stress: merge rate") ||
        !metric_close(unit.semantic.conditional_merge_error_severity, expected,
                      "A2 p1 stress: severity", 1.0e-12) ||
        !expect(unit.semantic.micro_impurity.support_count ==
                        unit.pair_counts.total_pairs &&
                    unit.semantic.macro_impurity.support_count == 1,
                "A2 p1 stress: supports wrong")) {
      return 1;
    }
  }

  // A2 p=2 stress independently exercises the stable linear-time moments path.
  {
    constexpr std::int64_t observations = 50'000;
    const auto truth =
        torch::arange(observations, torch::kLong).reshape({observations, 1});
    const auto predicted = torch::zeros({observations}, torch::kLong);
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {static_cast<double>(observations - 1)};
    options.power = 2;
    const auto &unit =
        only_unit(evaluate_clustering(predicted, truth, options));
    const auto n = static_cast<double>(observations);
    const auto expected = n * (n + 1.0) / (6.0 * (n - 1.0) * (n - 1.0));
    if (!metric_close(unit.semantic.micro_impurity, expected,
                      "A2 p2 stress: micro impurity", 1.0e-12) ||
        !metric_close(unit.semantic.macro_impurity, expected,
                      "A2 p2 stress: macro impurity", 1.0e-12) ||
        !metric_close(unit.semantic.conditional_merge_error_severity, expected,
                      "A2 p2 stress: severity", 1.0e-12)) {
      return 1;
    }
  }

  // A3 stress keeps N large while the dense assignment remains small. Costs
  // must come from contingency masses, not observation-pair materialization.
  {
    constexpr std::int64_t observations = 50'000;
    const auto truth_labels = torch::arange(observations, torch::kLong) % 2;
    const auto predicted = 1 - truth_labels;
    ClusteringEvaluationOptions options;
    options.semantic = SemanticEvaluationMode::Power;
    options.semantic_ranges = {1.0};
    options.alignment = AlignmentEvaluationMode::Both;
    const auto &unit = only_unit(evaluate_clustering(
        predicted, truth_labels.reshape({observations, 1}), options));
    if (!expect(unit.exact_alignment->mapping.predicted_to_truth_group ==
                        std::vector<std::int64_t>({1, 0}) &&
                    unit.semantic_alignment->mapping.predicted_to_truth_group ==
                        std::vector<std::int64_t>({1, 0}),
                "A3 stress: mapping wrong") ||
        !metric_close(unit.exact_alignment->matched_accuracy, 1.0,
                      "A3 stress: exact accuracy") ||
        !metric_close(unit.semantic_alignment->normalized_cost, 0.0,
                      "A3 stress: semantic cost") ||
        !expect(unit.semantic_alignment->normalized_cost.support_count ==
                    static_cast<std::uint64_t>(observations),
                "A3 stress: support wrong")) {
      return 1;
    }
  }

  std::cout << "clustering_evaluation tests passed\n";
  return 0;
}
