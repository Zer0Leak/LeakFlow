#include "leakflow/ml/clustering_evaluation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace leakflow::ml {
namespace {

constexpr std::size_t kExactMetricCount = 10;
constexpr std::array<ClusteringMetricDescriptor, 20> kMetricDescriptors{{
    {ClusteringMetricId::AdjustedRandIndex, "adjusted_rand_index",
     MetricFamily::Exact, MetricDirection::HigherIsBetter,
     MetricAveraging::None},
    {ClusteringMetricId::AdjustedMutualInformation,
     "adjusted_mutual_information", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::None},
    {ClusteringMetricId::Homogeneity, "homogeneity", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::None},
    {ClusteringMetricId::Completeness, "completeness", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::None},
    {ClusteringMetricId::VMeasure, "v_measure", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::None},
    {ClusteringMetricId::Purity, "purity", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::Micro},
    {ClusteringMetricId::PairPrecision, "pair_precision", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::Micro},
    {ClusteringMetricId::PairRecall, "pair_recall", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::Micro},
    {ClusteringMetricId::PairF1, "pair_f1", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::Micro},
    {ClusteringMetricId::NormalizedMutualInformation,
     "normalized_mutual_information", MetricFamily::Exact,
     MetricDirection::HigherIsBetter, MetricAveraging::None},
    {ClusteringMetricId::SemanticImpurityMicro, "semantic_impurity_micro",
     MetricFamily::Semantic, MetricDirection::LowerIsBetter,
     MetricAveraging::Micro},
    {ClusteringMetricId::SemanticImpurityMacro, "semantic_impurity_macro",
     MetricFamily::Semantic, MetricDirection::LowerIsBetter,
     MetricAveraging::Macro},
    {ClusteringMetricId::MergeErrorRate, "merge_error_rate",
     MetricFamily::Semantic, MetricDirection::LowerIsBetter,
     MetricAveraging::Micro},
    {ClusteringMetricId::ConditionalMergeErrorSeverity,
     "conditional_merge_error_severity", MetricFamily::Semantic,
     MetricDirection::LowerIsBetter, MetricAveraging::Micro},
    {ClusteringMetricId::SemanticImpurityDimensionMicro,
     "semantic_impurity_dimension_micro", MetricFamily::Semantic,
     MetricDirection::LowerIsBetter, MetricAveraging::PerDimension},
    {ClusteringMetricId::SemanticImpurityDimensionMacro,
     "semantic_impurity_dimension_macro", MetricFamily::Semantic,
     MetricDirection::LowerIsBetter, MetricAveraging::PerDimension},
    {ClusteringMetricId::SemanticImpurityPerCluster,
     "semantic_impurity_per_cluster", MetricFamily::Semantic,
     MetricDirection::LowerIsBetter, MetricAveraging::PerCluster},
    {ClusteringMetricId::FragmentationMicro, "fragmentation_micro",
     MetricFamily::Fragmentation, MetricDirection::LowerIsBetter,
     MetricAveraging::Micro},
    {ClusteringMetricId::FragmentationMacro, "fragmentation_macro",
     MetricFamily::Fragmentation, MetricDirection::LowerIsBetter,
     MetricAveraging::Macro},
    {ClusteringMetricId::FragmentationPerGroup, "fragmentation_per_group",
     MetricFamily::Fragmentation, MetricDirection::LowerIsBetter,
     MetricAveraging::PerGroup},
}};

class CompensatedSum {
public:
  void add(long double value) {
    const auto adjusted = value - correction_;
    const auto next = sum_ + adjusted;
    correction_ = (next - sum_) - adjusted;
    sum_ = next;
  }

  [[nodiscard]] long double value() const noexcept { return sum_; }

private:
  long double sum_ = 0.0L;
  long double correction_ = 0.0L;
};

[[nodiscard]] std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                                        const char *what) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string("clustering evaluation ") + what +
                              " overflow");
  }
  return left + right;
}

[[nodiscard]] std::uint64_t checked_comb2(std::uint64_t count) {
  if (count < 2) {
    return 0;
  }
  auto left = count;
  auto right = count - 1;
  if ((left & 1U) == 0U) {
    left /= 2;
  } else {
    right /= 2;
  }
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::overflow_error("clustering evaluation pair count overflow");
  }
  return left * right;
}

[[nodiscard]] MetricValue defined_metric(ClusteringMetricId metric,
                                         double value,
                                         std::uint64_t support_count) {
  return {metric, value, support_count, MetricUndefinedReason::None};
}

[[nodiscard]] MetricValue undefined_metric(ClusteringMetricId metric,
                                           std::uint64_t support_count,
                                           MetricUndefinedReason reason) {
  return {metric, std::nullopt, support_count, reason};
}

[[nodiscard]] long double entropy(const std::vector<std::uint64_t> &marginal,
                                  std::uint64_t observations) {
  CompensatedSum result;
  const auto n = static_cast<long double>(observations);
  for (const auto count : marginal) {
    if (count == 0) {
      continue;
    }
    const auto probability = static_cast<long double>(count) / n;
    result.add(-probability * std::log(probability));
  }
  return result.value();
}

using ContingencyCounts =
    std::map<std::pair<std::int64_t, std::int64_t>, std::uint64_t>;

[[nodiscard]] long double
mutual_information(const ContingencyCounts &contingency,
                   const std::vector<std::uint64_t> &truth_marginal,
                   const std::vector<std::uint64_t> &predicted_marginal,
                   std::uint64_t observations) {
  CompensatedSum result;
  const auto n = static_cast<long double>(observations);
  for (const auto &[cell, count] : contingency) {
    const auto [truth_index, predicted_index] = cell;
    const auto truth_count =
        truth_marginal[static_cast<std::size_t>(truth_index)];
    const auto predicted_count =
        predicted_marginal[static_cast<std::size_t>(predicted_index)];
    const auto cell_count = static_cast<long double>(count);
    const auto probability = cell_count / n;
    const auto ratio =
        (n * cell_count) / (static_cast<long double>(truth_count) *
                            static_cast<long double>(predicted_count));
    result.add(probability * std::log(ratio));
  }
  return std::max(0.0L, result.value());
}

[[nodiscard]] long double log_choose(std::uint64_t n, std::uint64_t k) {
  if (k > n) {
    return -std::numeric_limits<long double>::infinity();
  }
  return std::lgamma(static_cast<long double>(n) + 1.0L) -
         std::lgamma(static_cast<long double>(k) + 1.0L) -
         std::lgamma(static_cast<long double>(n - k) + 1.0L);
}

[[nodiscard]] long double hypergeometric_log_probability(
    std::uint64_t observations, std::uint64_t truth_count,
    std::uint64_t predicted_count, std::uint64_t overlap) {
  return log_choose(truth_count, overlap) +
         log_choose(observations - truth_count, predicted_count - overlap) -
         log_choose(observations, predicted_count);
}

[[nodiscard]] long double expected_mutual_information(
    const std::vector<std::uint64_t> &truth_marginal,
    const std::vector<std::uint64_t> &predicted_marginal,
    std::uint64_t observations) {
  const auto marginal_pair_contribution =
      [observations](std::uint64_t truth_count, std::uint64_t predicted_count) {
        CompensatedSum contribution;
        const auto n_ld = static_cast<long double>(observations);
        const auto lower = truth_count > observations - predicted_count
                               ? truth_count - (observations - predicted_count)
                               : 0;
        const auto upper = std::min(truth_count, predicted_count);

        long double max_log_probability =
            -std::numeric_limits<long double>::infinity();
        for (auto overlap = lower;; ++overlap) {
          max_log_probability = std::max(
              max_log_probability,
              hypergeometric_log_probability(observations, truth_count,
                                             predicted_count, overlap));
          if (overlap == upper) {
            break;
          }
        }

        CompensatedSum normalizer;
        for (auto overlap = lower;; ++overlap) {
          const auto log_probability = hypergeometric_log_probability(
              observations, truth_count, predicted_count, overlap);
          normalizer.add(std::exp(log_probability - max_log_probability));
          if (overlap == upper) {
            break;
          }
        }
        const auto normalization = normalizer.value();
        if (!(normalization > 0.0L) || !std::isfinite(normalization)) {
          throw std::runtime_error(
              "clustering evaluation AMI probability normalization failed");
        }

        const auto positive_lower = std::max<std::uint64_t>(1, lower);
        if (positive_lower > upper) {
          return 0.0L;
        }
        for (auto overlap = positive_lower;; ++overlap) {
          const auto log_probability = hypergeometric_log_probability(
              observations, truth_count, predicted_count, overlap);
          const auto probability =
              std::exp(log_probability - max_log_probability) / normalization;
          const auto overlap_ld = static_cast<long double>(overlap);
          const auto information =
              (overlap_ld / n_ld) *
              std::log((n_ld * overlap_ld) /
                       (static_cast<long double>(truth_count) *
                        static_cast<long double>(predicted_count)));
          contribution.add(probability * information);
          if (overlap == upper) {
            break;
          }
        }
        return contribution.value();
      };

  // Many high-cardinality partitions repeat the same marginal sizes. Evaluate
  // each distinct size pair once, then scale by its multiplicity, so expected
  // MI does not require a raw G*K loop in those cases.
  std::map<std::uint64_t, std::uint64_t> truth_size_frequency;
  std::map<std::uint64_t, std::uint64_t> predicted_size_frequency;
  for (const auto count : truth_marginal) {
    auto &frequency = truth_size_frequency[count];
    frequency = checked_add(frequency, 1, "truth marginal frequency");
  }
  for (const auto count : predicted_marginal) {
    auto &frequency = predicted_size_frequency[count];
    frequency = checked_add(frequency, 1, "predicted marginal frequency");
  }

  CompensatedSum expected;
  for (const auto &[truth_count, truth_frequency] : truth_size_frequency) {
    for (const auto &[predicted_count, predicted_frequency] :
         predicted_size_frequency) {
      const auto multiplicity = static_cast<long double>(truth_frequency) *
                                static_cast<long double>(predicted_frequency);
      expected.add(marginal_pair_contribution(truth_count, predicted_count) *
                   multiplicity);
    }
  }
  return expected.value();
}

[[nodiscard]] double adjusted_mutual_information(
    long double mutual, long double truth_entropy,
    long double predicted_entropy,
    const std::vector<std::uint64_t> &truth_marginal,
    const std::vector<std::uint64_t> &predicted_marginal,
    std::uint64_t observations) {
  if (truth_marginal.size() == 1 && predicted_marginal.size() == 1) {
    return 1.0;
  }
  if (truth_marginal.size() == 1 || predicted_marginal.size() == 1) {
    return 0.0;
  }
  if (truth_marginal.size() == observations &&
      predicted_marginal.size() == observations) {
    return 1.0;
  }

  const auto expected = expected_mutual_information(
      truth_marginal, predicted_marginal, observations);
  const auto normalizer = 0.5L * (truth_entropy + predicted_entropy);
  auto denominator = static_cast<double>(normalizer - expected);
  auto numerator = static_cast<double>(mutual - expected);
  const auto epsilon = std::numeric_limits<double>::epsilon();
  denominator = denominator < 0.0 ? std::min(denominator, -epsilon)
                                  : std::max(denominator, epsilon);
  numerator = numerator < 0.0 ? std::min(numerator, -epsilon)
                              : std::max(numerator, epsilon);
  return numerator / denominator;
}

[[nodiscard]] ExactClusteringMetrics
compute_exact_metrics(const ContingencyCounts &contingency,
                      const std::vector<std::uint64_t> &truth_marginal,
                      const std::vector<std::uint64_t> &predicted_marginal,
                      const PairCounts &pairs, std::uint64_t observations) {
  const auto mutual = mutual_information(contingency, truth_marginal,
                                         predicted_marginal, observations);
  const auto truth_entropy = entropy(truth_marginal, observations);
  const auto predicted_entropy = entropy(predicted_marginal, observations);

  const auto homogeneity = truth_marginal.size() == 1
                               ? 1.0
                               : static_cast<double>(mutual / truth_entropy);
  const auto completeness =
      predicted_marginal.size() == 1
          ? 1.0
          : static_cast<double>(mutual / predicted_entropy);
  const auto v_measure =
      homogeneity + completeness > 0.0
          ? 2.0 * homogeneity * completeness / (homogeneity + completeness)
          : 0.0;

  double nmi = 0.0;
  if (truth_marginal.size() == 1 && predicted_marginal.size() == 1) {
    nmi = 1.0;
  } else if (truth_marginal.size() != 1 && predicted_marginal.size() != 1) {
    nmi = static_cast<double>(2.0L * mutual /
                              (truth_entropy + predicted_entropy));
  }

  std::uint64_t purity_numerator = 0;
  std::vector<std::uint64_t> cluster_max(predicted_marginal.size(), 0);
  for (const auto &[cell, count] : contingency) {
    const auto predicted_index = static_cast<std::size_t>(cell.second);
    cluster_max[predicted_index] =
        std::max(cluster_max[predicted_index], count);
  }
  for (const auto count : cluster_max) {
    purity_numerator = checked_add(purity_numerator, count, "purity numerator");
  }
  const auto purity =
      static_cast<double>(static_cast<long double>(purity_numerator) /
                          static_cast<long double>(observations));

  double ari = 1.0;
  if (pairs.false_positive != 0 || pairs.false_negative != 0) {
    const auto tp = static_cast<long double>(pairs.true_positive);
    const auto fp = static_cast<long double>(pairs.false_positive);
    const auto fn = static_cast<long double>(pairs.false_negative);
    const auto tn = static_cast<long double>(pairs.true_negative);
    const auto numerator = 2.0L * (tp * tn - fn * fp);
    const auto denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn);
    ari = static_cast<double>(numerator / denominator);
  }

  ExactClusteringMetrics result;
  result.adjusted_rand_index =
      defined_metric(ClusteringMetricId::AdjustedRandIndex, ari, observations);
  result.adjusted_mutual_information =
      defined_metric(ClusteringMetricId::AdjustedMutualInformation,
                     adjusted_mutual_information(
                         mutual, truth_entropy, predicted_entropy,
                         truth_marginal, predicted_marginal, observations),
                     observations);
  result.homogeneity = defined_metric(ClusteringMetricId::Homogeneity,
                                      homogeneity, observations);
  result.completeness = defined_metric(ClusteringMetricId::Completeness,
                                       completeness, observations);
  result.v_measure =
      defined_metric(ClusteringMetricId::VMeasure, v_measure, observations);
  result.purity =
      defined_metric(ClusteringMetricId::Purity, purity, observations);
  result.normalized_mutual_information = defined_metric(
      ClusteringMetricId::NormalizedMutualInformation, nmi, observations);

  if (pairs.predicted_within_cluster_pairs == 0) {
    result.pair_precision =
        undefined_metric(ClusteringMetricId::PairPrecision, 0,
                         MetricUndefinedReason::NoPredictedWithinClusterPairs);
  } else {
    result.pair_precision = defined_metric(
        ClusteringMetricId::PairPrecision,
        static_cast<double>(pairs.true_positive) /
            static_cast<double>(pairs.predicted_within_cluster_pairs),
        pairs.predicted_within_cluster_pairs);
  }

  if (pairs.true_within_group_pairs == 0) {
    result.pair_recall =
        undefined_metric(ClusteringMetricId::PairRecall, 0,
                         MetricUndefinedReason::NoTrueWithinGroupPairs);
  } else {
    result.pair_recall =
        defined_metric(ClusteringMetricId::PairRecall,
                       static_cast<double>(pairs.true_positive) /
                           static_cast<double>(pairs.true_within_group_pairs),
                       pairs.true_within_group_pairs);
  }

  const auto pair_f1_support = checked_add(
      checked_add(checked_add(pairs.true_positive, pairs.true_positive,
                              "pair F1 support"),
                  pairs.false_positive, "pair F1 support"),
      pairs.false_negative, "pair F1 support");
  if (!result.pair_precision.defined() || !result.pair_recall.defined()) {
    result.pair_f1 =
        undefined_metric(ClusteringMetricId::PairF1, pair_f1_support,
                         MetricUndefinedReason::DependentMetricUndefined);
  } else {
    const auto numerator = 2.0L * static_cast<long double>(pairs.true_positive);
    const auto denominator = numerator +
                             static_cast<long double>(pairs.false_positive) +
                             static_cast<long double>(pairs.false_negative);
    result.pair_f1 = defined_metric(
        ClusteringMetricId::PairF1,
        denominator > 0.0L ? static_cast<double>(numerator / denominator) : 0.0,
        pair_f1_support);
  }

  return result;
}

[[nodiscard]] FragmentationClusteringMetrics
compute_fragmentation_metrics(const ContingencyCounts &contingency,
                              const std::vector<std::uint64_t> &truth_marginal,
                              const PairCounts &pairs,
                              ClusteringEvaluationDetail detail) {
  FragmentationClusteringMetrics result;
  if (pairs.true_within_group_pairs == 0) {
    result.micro =
        undefined_metric(ClusteringMetricId::FragmentationMicro, 0,
                         MetricUndefinedReason::NoTrueWithinGroupPairs);
  } else {
    result.micro =
        defined_metric(ClusteringMetricId::FragmentationMicro,
                       static_cast<double>(pairs.false_negative) /
                           static_cast<double>(pairs.true_within_group_pairs),
                       pairs.true_within_group_pairs);
  }

  std::vector<std::uint64_t> preserved_pairs(truth_marginal.size(), 0);
  for (const auto &[cell, count] : contingency) {
    const auto truth_index = static_cast<std::size_t>(cell.first);
    preserved_pairs[truth_index] =
        checked_add(preserved_pairs[truth_index], checked_comb2(count),
                    "truth-group preserved pairs");
  }

  if (detail == ClusteringEvaluationDetail::Full) {
    result.group_details.emplace();
    result.group_details->reserve(truth_marginal.size());
  }

  CompensatedSum macro_sum;
  std::uint64_t eligible_groups = 0;
  for (std::size_t truth_index = 0; truth_index < truth_marginal.size();
       ++truth_index) {
    const auto group_pairs = checked_comb2(truth_marginal[truth_index]);
    MetricValue fragmentation;
    if (group_pairs == 0) {
      fragmentation =
          undefined_metric(ClusteringMetricId::FragmentationPerGroup, 0,
                           MetricUndefinedReason::NoTrueWithinGroupPairs);
    } else {
      if (preserved_pairs[truth_index] > group_pairs) {
        throw std::logic_error(
            "clustering evaluation truth-group pair counts are inconsistent");
      }
      const auto fragmented_pairs = group_pairs - preserved_pairs[truth_index];
      const auto value = static_cast<double>(fragmented_pairs) /
                         static_cast<double>(group_pairs);
      fragmentation = defined_metric(ClusteringMetricId::FragmentationPerGroup,
                                     value, group_pairs);
      macro_sum.add(static_cast<long double>(value));
      eligible_groups =
          checked_add(eligible_groups, 1, "eligible truth groups");
    }

    if (result.group_details.has_value()) {
      result.group_details->push_back({static_cast<std::int64_t>(truth_index),
                                       truth_marginal[truth_index],
                                       std::move(fragmentation)});
    }
  }

  if (eligible_groups == 0) {
    result.macro =
        undefined_metric(ClusteringMetricId::FragmentationMacro, 0,
                         MetricUndefinedReason::NoEligibleTruthGroups);
  } else {
    result.macro = defined_metric(
        ClusteringMetricId::FragmentationMacro,
        static_cast<double>(macro_sum.value() /
                            static_cast<long double>(eligible_groups)),
        eligible_groups);
  }
  return result;
}

struct SemanticPairSums {
  std::vector<long double> aggregate_by_cluster;
  std::vector<std::vector<long double>> dimension_by_cluster;
};

class RunningVariance {
public:
  void add(long double value) {
    count_ = checked_add(count_, 1, "semantic moment count");
    const auto delta = value - mean_;
    mean_ += delta / static_cast<long double>(count_);
    second_moment_.add(delta * (value - mean_));
  }

  [[nodiscard]] long double pair_sum() const {
    return static_cast<long double>(count_) * second_moment_.value();
  }

private:
  std::uint64_t count_ = 0;
  long double mean_ = 0.0L;
  CompensatedSum second_moment_;
};

[[nodiscard]] long double validated_pair_sum(long double value,
                                             std::uint64_t pair_count) {
  const auto upper = static_cast<long double>(pair_count);
  const auto tolerance = 64.0L * std::numeric_limits<long double>::epsilon() *
                         std::max(1.0L, upper);
  if (value < -tolerance || value > upper + tolerance ||
      !std::isfinite(value)) {
    throw std::runtime_error(
        "clustering evaluation semantic pair aggregation failed");
  }
  return std::clamp(value, 0.0L, upper);
}

template <typename Scalar>
[[nodiscard]] long double nonnegative_coordinate_difference(Scalar high,
                                                            Scalar low) {
  if constexpr (std::is_floating_point_v<Scalar>) {
    return static_cast<long double>(high) - static_cast<long double>(low);
  } else if constexpr (std::is_unsigned_v<Scalar>) {
    return static_cast<long double>(high - low);
  } else {
    using Unsigned = std::make_unsigned_t<Scalar>;
    return static_cast<long double>(static_cast<Unsigned>(high) -
                                    static_cast<Unsigned>(low));
  }
}

template <typename Scalar>
[[nodiscard]] std::vector<Scalar>
semantic_dimension_minima(const torch::TensorAccessor<Scalar, 2> &truth,
                          std::int64_t observations, std::int64_t dimensions,
                          const ClusteringEvaluationOptions &options,
                          std::int64_t unit_index) {
  std::vector<Scalar> minima(static_cast<std::size_t>(dimensions));
  std::vector<Scalar> maxima(static_cast<std::size_t>(dimensions));
  for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
    const auto value = truth[0][dimension];
    minima[static_cast<std::size_t>(dimension)] = value;
    maxima[static_cast<std::size_t>(dimension)] = value;
  }
  for (std::int64_t observation = 1; observation < observations;
       ++observation) {
    for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
      const auto value = truth[observation][dimension];
      const auto index = static_cast<std::size_t>(dimension);
      minima[index] = std::min(minima[index], value);
      maxima[index] = std::max(maxima[index], value);
    }
  }
  for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
    const auto index = static_cast<std::size_t>(dimension);
    const auto span =
        nonnegative_coordinate_difference(maxima[index], minima[index]);
    const auto range = static_cast<long double>(options.semantic_ranges[index]);
    if (span > range) {
      throw std::invalid_argument(
          "clustering evaluation semantic values exceed configured range at "
          "unit " +
          std::to_string(unit_index) + ", dimension " +
          std::to_string(dimension));
    }
  }
  return minima;
}

template <typename Scalar>
[[nodiscard]] SemanticPairSums compute_semantic_pair_sums(
    const torch::TensorAccessor<Scalar, 2> &truth,
    const std::vector<std::int64_t> &predicted_assignment,
    const std::vector<std::uint64_t> &predicted_marginal,
    const ClusteringEvaluationOptions &options, std::int64_t observations,
    std::int64_t dimensions, std::int64_t unit_index) {
  const auto clusters = predicted_marginal.size();
  const auto dimension_count = static_cast<std::size_t>(dimensions);
  const auto minima = semantic_dimension_minima(truth, observations, dimensions,
                                                options, unit_index);

  SemanticPairSums result;
  result.aggregate_by_cluster.assign(clusters, 0.0L);
  result.dimension_by_cluster.assign(
      clusters, std::vector<long double>(dimension_count, 0.0L));

  if (options.power == 2) {
    std::vector<RunningVariance> moments(clusters * dimension_count);
    for (std::int64_t observation = 0; observation < observations;
         ++observation) {
      const auto cluster =
          static_cast<std::size_t>(predicted_assignment[observation]);
      for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
        const auto dimension_index = static_cast<std::size_t>(dimension);
        const auto flat_index = cluster * dimension_count + dimension_index;
        const auto value =
            nonnegative_coordinate_difference(truth[observation][dimension],
                                              minima[dimension_index]) /
            static_cast<long double>(options.semantic_ranges[dimension_index]);
        moments[flat_index].add(value);
      }
    }
    for (std::size_t cluster = 0; cluster < clusters; ++cluster) {
      const auto pair_count = checked_comb2(predicted_marginal[cluster]);
      for (std::size_t dimension = 0; dimension < dimension_count;
           ++dimension) {
        const auto flat_index = cluster * dimension_count + dimension;
        result.dimension_by_cluster[cluster][dimension] =
            validated_pair_sum(moments[flat_index].pair_sum(), pair_count);
      }
    }
  } else {
    std::vector<std::vector<long double>> values(clusters * dimension_count);
    for (std::int64_t observation = 0; observation < observations;
         ++observation) {
      const auto cluster =
          static_cast<std::size_t>(predicted_assignment[observation]);
      for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
        const auto dimension_index = static_cast<std::size_t>(dimension);
        values[cluster * dimension_count + dimension_index].push_back(
            nonnegative_coordinate_difference(truth[observation][dimension],
                                              minima[dimension_index]) /
            static_cast<long double>(options.semantic_ranges[dimension_index]));
      }
    }
    for (std::size_t cluster = 0; cluster < clusters; ++cluster) {
      const auto pair_count = checked_comb2(predicted_marginal[cluster]);
      for (std::size_t dimension = 0; dimension < dimension_count;
           ++dimension) {
        auto &coordinate_values = values[cluster * dimension_count + dimension];
        std::sort(coordinate_values.begin(), coordinate_values.end());
        CompensatedSum prefix;
        CompensatedSum pair_sum;
        for (std::size_t index = 0; index < coordinate_values.size(); ++index) {
          const auto value = coordinate_values[index];
          pair_sum.add(static_cast<long double>(index) * value -
                       prefix.value());
          prefix.add(value);
        }
        result.dimension_by_cluster[cluster][dimension] =
            validated_pair_sum(pair_sum.value(), pair_count);
      }
    }
  }

  const auto max_weight = *std::max_element(options.semantic_weights.begin(),
                                            options.semantic_weights.end());
  CompensatedSum scaled_weight_sum;
  for (const auto weight : options.semantic_weights) {
    scaled_weight_sum.add(static_cast<long double>(weight / max_weight));
  }
  for (std::size_t cluster = 0; cluster < clusters; ++cluster) {
    CompensatedSum weighted_sum;
    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
      weighted_sum.add(static_cast<long double>(
                           options.semantic_weights[dimension] / max_weight) *
                       result.dimension_by_cluster[cluster][dimension]);
    }
    result.aggregate_by_cluster[cluster] =
        validated_pair_sum(weighted_sum.value() / scaled_weight_sum.value(),
                           checked_comb2(predicted_marginal[cluster]));
  }
  return result;
}

[[nodiscard]] SemanticClusteringMetrics
semantic_disabled_metrics(const std::vector<std::uint64_t> &predicted_marginal,
                          const PairCounts &pairs, std::int64_t dimensions,
                          ClusteringEvaluationDetail detail) {
  SemanticClusteringMetrics result;
  std::uint64_t eligible_clusters = 0;
  for (const auto count : predicted_marginal) {
    if (count >= 2) {
      eligible_clusters =
          checked_add(eligible_clusters, 1, "eligible predicted clusters");
    }
  }
  result.micro_impurity =
      undefined_metric(ClusteringMetricId::SemanticImpurityMicro,
                       pairs.predicted_within_cluster_pairs,
                       MetricUndefinedReason::SemanticDisabled);
  result.macro_impurity = undefined_metric(
      ClusteringMetricId::SemanticImpurityMacro, eligible_clusters,
      MetricUndefinedReason::SemanticDisabled);
  result.merge_error_rate = undefined_metric(
      ClusteringMetricId::MergeErrorRate, pairs.predicted_within_cluster_pairs,
      MetricUndefinedReason::SemanticDisabled);
  result.conditional_merge_error_severity = undefined_metric(
      ClusteringMetricId::ConditionalMergeErrorSeverity, pairs.false_positive,
      MetricUndefinedReason::SemanticDisabled);
  result.dimensions.reserve(static_cast<std::size_t>(dimensions));
  for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
    result.dimensions.push_back(
        {dimension,
         undefined_metric(ClusteringMetricId::SemanticImpurityDimensionMicro,
                          pairs.predicted_within_cluster_pairs,
                          MetricUndefinedReason::SemanticDisabled),
         undefined_metric(ClusteringMetricId::SemanticImpurityDimensionMacro,
                          eligible_clusters,
                          MetricUndefinedReason::SemanticDisabled)});
  }
  if (detail == ClusteringEvaluationDetail::Full) {
    result.cluster_details.emplace();
    result.cluster_details->reserve(predicted_marginal.size());
    for (std::size_t cluster = 0; cluster < predicted_marginal.size();
         ++cluster) {
      const auto support = checked_comb2(predicted_marginal[cluster]);
      result.cluster_details->push_back(
          {static_cast<std::int64_t>(cluster), predicted_marginal[cluster],
           undefined_metric(ClusteringMetricId::SemanticImpurityPerCluster,
                            support, MetricUndefinedReason::SemanticDisabled)});
    }
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] SemanticClusteringMetrics
compute_semantic_metrics(const torch::TensorAccessor<Scalar, 2> &truth,
                         const std::vector<std::int64_t> &predicted_assignment,
                         const std::vector<std::uint64_t> &predicted_marginal,
                         const PairCounts &pairs,
                         const ClusteringEvaluationOptions &options,
                         std::int64_t observations, std::int64_t dimensions,
                         std::int64_t unit_index) {
  if (options.semantic == SemanticEvaluationMode::Off) {
    return semantic_disabled_metrics(predicted_marginal, pairs, dimensions,
                                     options.detail);
  }

  const auto pair_sums = compute_semantic_pair_sums(
      truth, predicted_assignment, predicted_marginal, options, observations,
      dimensions, unit_index);
  SemanticClusteringMetrics result;
  const auto dimension_count = static_cast<std::size_t>(dimensions);
  std::vector<CompensatedSum> dimension_totals(dimension_count);
  std::vector<CompensatedSum> dimension_macro_sums(dimension_count);
  CompensatedSum aggregate_total;
  CompensatedSum aggregate_macro_sum;
  std::uint64_t eligible_clusters = 0;

  if (options.detail == ClusteringEvaluationDetail::Full) {
    result.cluster_details.emplace();
    result.cluster_details->reserve(predicted_marginal.size());
  }
  for (std::size_t cluster = 0; cluster < predicted_marginal.size();
       ++cluster) {
    const auto cluster_pairs = checked_comb2(predicted_marginal[cluster]);
    aggregate_total.add(pair_sums.aggregate_by_cluster[cluster]);
    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
      dimension_totals[dimension].add(
          pair_sums.dimension_by_cluster[cluster][dimension]);
    }

    MetricValue cluster_impurity;
    if (cluster_pairs == 0) {
      cluster_impurity = undefined_metric(
          ClusteringMetricId::SemanticImpurityPerCluster, 0,
          MetricUndefinedReason::NoPredictedWithinClusterPairs);
    } else {
      const auto impurity = pair_sums.aggregate_by_cluster[cluster] /
                            static_cast<long double>(cluster_pairs);
      cluster_impurity =
          defined_metric(ClusteringMetricId::SemanticImpurityPerCluster,
                         static_cast<double>(impurity), cluster_pairs);
      aggregate_macro_sum.add(impurity);
      for (std::size_t dimension = 0; dimension < dimension_count;
           ++dimension) {
        dimension_macro_sums[dimension].add(
            pair_sums.dimension_by_cluster[cluster][dimension] /
            static_cast<long double>(cluster_pairs));
      }
      eligible_clusters =
          checked_add(eligible_clusters, 1, "eligible predicted clusters");
    }
    if (result.cluster_details.has_value()) {
      result.cluster_details->push_back({static_cast<std::int64_t>(cluster),
                                         predicted_marginal[cluster],
                                         std::move(cluster_impurity)});
    }
  }

  // Equal truth vectors contribute exactly zero, so the accumulated semantic
  // numerator is supported only by false-positive (merge-error) pairs.
  const auto semantic_error_total =
      validated_pair_sum(aggregate_total.value(), pairs.false_positive);
  std::vector<long double> dimension_error_totals(dimension_count, 0.0L);
  for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
    dimension_error_totals[dimension] = validated_pair_sum(
        dimension_totals[dimension].value(), pairs.false_positive);
  }

  if (pairs.predicted_within_cluster_pairs == 0) {
    result.micro_impurity =
        undefined_metric(ClusteringMetricId::SemanticImpurityMicro, 0,
                         MetricUndefinedReason::NoPredictedWithinClusterPairs);
    result.merge_error_rate =
        undefined_metric(ClusteringMetricId::MergeErrorRate, 0,
                         MetricUndefinedReason::NoPredictedWithinClusterPairs);
  } else {
    result.micro_impurity = defined_metric(
        ClusteringMetricId::SemanticImpurityMicro,
        static_cast<double>(
            semantic_error_total /
            static_cast<long double>(pairs.predicted_within_cluster_pairs)),
        pairs.predicted_within_cluster_pairs);
    result.merge_error_rate = defined_metric(
        ClusteringMetricId::MergeErrorRate,
        static_cast<double>(pairs.false_positive) /
            static_cast<double>(pairs.predicted_within_cluster_pairs),
        pairs.predicted_within_cluster_pairs);
  }

  if (eligible_clusters == 0) {
    result.macro_impurity =
        undefined_metric(ClusteringMetricId::SemanticImpurityMacro, 0,
                         MetricUndefinedReason::NoEligiblePredictedClusters);
  } else {
    result.macro_impurity = defined_metric(
        ClusteringMetricId::SemanticImpurityMacro,
        static_cast<double>(aggregate_macro_sum.value() /
                            static_cast<long double>(eligible_clusters)),
        eligible_clusters);
  }

  if (pairs.false_positive == 0) {
    result.conditional_merge_error_severity =
        undefined_metric(ClusteringMetricId::ConditionalMergeErrorSeverity, 0,
                         MetricUndefinedReason::NoMergeErrorPairs);
  } else {
    result.conditional_merge_error_severity = defined_metric(
        ClusteringMetricId::ConditionalMergeErrorSeverity,
        static_cast<double>(semantic_error_total /
                            static_cast<long double>(pairs.false_positive)),
        pairs.false_positive);
  }

  result.dimensions.reserve(dimension_count);
  for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
    MetricValue micro;
    if (pairs.predicted_within_cluster_pairs == 0) {
      micro = undefined_metric(
          ClusteringMetricId::SemanticImpurityDimensionMicro, 0,
          MetricUndefinedReason::NoPredictedWithinClusterPairs);
    } else {
      micro = defined_metric(
          ClusteringMetricId::SemanticImpurityDimensionMicro,
          static_cast<double>(
              dimension_error_totals[dimension] /
              static_cast<long double>(pairs.predicted_within_cluster_pairs)),
          pairs.predicted_within_cluster_pairs);
    }
    MetricValue macro;
    if (eligible_clusters == 0) {
      macro = undefined_metric(
          ClusteringMetricId::SemanticImpurityDimensionMacro, 0,
          MetricUndefinedReason::NoEligiblePredictedClusters);
    } else {
      macro = defined_metric(
          ClusteringMetricId::SemanticImpurityDimensionMacro,
          static_cast<double>(dimension_macro_sums[dimension].value() /
                              static_cast<long double>(eligible_clusters)),
          eligible_clusters);
    }
    result.dimensions.push_back({static_cast<std::int64_t>(dimension),
                                 std::move(micro), std::move(macro)});
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] std::vector<Scalar>
semantic_key(const torch::TensorAccessor<Scalar, 2> &truth,
             std::int64_t observation, std::int64_t dimensions) {
  std::vector<Scalar> key;
  key.reserve(static_cast<std::size_t>(dimensions));
  for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
    key.push_back(truth[observation][dimension]);
  }
  return key;
}

template <typename Scalar>
[[nodiscard]] ClusteringEvaluationUnitResult evaluate_unit(
    const torch::Tensor &predicted, const torch::Tensor &truth_key_values,
    const torch::Tensor &truth_output_values,
    const ClusteringEvaluationOptions &options, std::int64_t unit_index) {
  const auto observations = predicted.size(0);
  const auto dimensions = truth_key_values.size(1);
  const auto predicted_access = predicted.accessor<std::int64_t, 1>();
  const auto truth_access = truth_key_values.accessor<Scalar, 2>();

  std::set<std::int64_t> unique_predicted;
  std::map<std::vector<Scalar>, std::int64_t> unique_truth;
  for (std::int64_t observation = 0; observation < observations;
       ++observation) {
    unique_predicted.insert(predicted_access[observation]);
    auto key = semantic_key(truth_access, observation, dimensions);
    unique_truth.try_emplace(std::move(key), observation);
  }

  std::map<std::int64_t, std::int64_t> predicted_index;
  std::vector<std::int64_t> predicted_ids;
  predicted_ids.reserve(unique_predicted.size());
  for (const auto id : unique_predicted) {
    const auto index = static_cast<std::int64_t>(predicted_ids.size());
    predicted_index.emplace(id, index);
    predicted_ids.push_back(id);
  }

  std::map<std::vector<Scalar>, std::int64_t> truth_index;
  std::vector<std::int64_t> truth_representative_rows;
  truth_representative_rows.reserve(unique_truth.size());
  for (const auto &[key, representative_row] : unique_truth) {
    const auto index =
        static_cast<std::int64_t>(truth_representative_rows.size());
    truth_index.emplace(key, index);
    truth_representative_rows.push_back(representative_row);
  }

  std::vector<std::uint64_t> truth_marginal(unique_truth.size(), 0);
  std::vector<std::uint64_t> predicted_marginal(unique_predicted.size(), 0);
  std::vector<std::int64_t> predicted_assignment(
      static_cast<std::size_t>(observations));
  ContingencyCounts contingency;
  for (std::int64_t observation = 0; observation < observations;
       ++observation) {
    const auto predicted_dense =
        predicted_index.at(predicted_access[observation]);
    const auto truth_dense =
        truth_index.at(semantic_key(truth_access, observation, dimensions));
    const auto truth_position = static_cast<std::size_t>(truth_dense);
    const auto predicted_position = static_cast<std::size_t>(predicted_dense);
    predicted_assignment[static_cast<std::size_t>(observation)] =
        predicted_dense;
    truth_marginal[truth_position] =
        checked_add(truth_marginal[truth_position], 1, "truth marginal");
    predicted_marginal[predicted_position] = checked_add(
        predicted_marginal[predicted_position], 1, "predicted marginal");
    auto &cell = contingency[{truth_dense, predicted_dense}];
    cell = checked_add(cell, 1, "contingency count");
  }

  PairCounts pairs;
  pairs.total_pairs = checked_comb2(static_cast<std::uint64_t>(observations));
  for (const auto &[cell, count] : contingency) {
    static_cast<void>(cell);
    pairs.true_positive = checked_add(pairs.true_positive, checked_comb2(count),
                                      "true positives");
  }
  for (const auto count : predicted_marginal) {
    pairs.predicted_within_cluster_pairs =
        checked_add(pairs.predicted_within_cluster_pairs, checked_comb2(count),
                    "predicted pair count");
  }
  for (const auto count : truth_marginal) {
    pairs.true_within_group_pairs =
        checked_add(pairs.true_within_group_pairs, checked_comb2(count),
                    "truth pair count");
  }
  pairs.false_positive =
      pairs.predicted_within_cluster_pairs - pairs.true_positive;
  pairs.false_negative = pairs.true_within_group_pairs - pairs.true_positive;
  const auto non_true_negative = checked_add(
      checked_add(pairs.true_positive, pairs.false_positive, "pair partition"),
      pairs.false_negative, "pair partition");
  if (non_true_negative > pairs.total_pairs) {
    throw std::logic_error(
        "clustering evaluation pair counts are inconsistent");
  }
  pairs.true_negative = pairs.total_pairs - non_true_negative;

  ClusteringEvaluationUnitResult result;
  result.observation_count = observations;
  result.truth_group_count = static_cast<std::int64_t>(truth_marginal.size());
  result.predicted_cluster_count =
      static_cast<std::int64_t>(predicted_marginal.size());
  result.pair_counts = pairs;
  result.exact =
      compute_exact_metrics(contingency, truth_marginal, predicted_marginal,
                            pairs, static_cast<std::uint64_t>(observations));
  result.semantic = compute_semantic_metrics(
      truth_access, predicted_assignment, predicted_marginal, pairs, options,
      observations, dimensions, unit_index);
  result.fragmentation = compute_fragmentation_metrics(
      contingency, truth_marginal, pairs, options.detail);

  if (options.detail == ClusteringEvaluationDetail::Full) {
    ClusteringPartitionDetail partition;
    partition.predicted_ids = torch::tensor(predicted_ids, torch::kLong);
    const auto representative_index =
        torch::tensor(truth_representative_rows, torch::kLong);
    partition.truth_vectors =
        truth_output_values.index_select(0, representative_index).contiguous();

    const auto nonzero = static_cast<std::int64_t>(contingency.size());
    partition.contingency.truth_group_indices =
        torch::empty({nonzero}, torch::kLong);
    partition.contingency.predicted_cluster_indices =
        torch::empty({nonzero}, torch::kLong);
    partition.contingency.counts = torch::empty({nonzero}, torch::kLong);
    partition.contingency.truth_group_count = result.truth_group_count;
    partition.contingency.predicted_cluster_count =
        result.predicted_cluster_count;
    auto truth_indices =
        partition.contingency.truth_group_indices.accessor<std::int64_t, 1>();
    auto predicted_indices = partition.contingency.predicted_cluster_indices
                                 .accessor<std::int64_t, 1>();
    auto counts = partition.contingency.counts.accessor<std::int64_t, 1>();
    std::int64_t offset = 0;
    for (const auto &[cell, count] : contingency) {
      truth_indices[offset] = cell.first;
      predicted_indices[offset] = cell.second;
      counts[offset] = static_cast<std::int64_t>(count);
      ++offset;
    }
    result.partition_detail = std::move(partition);
  }

  return result;
}

[[nodiscard]] std::vector<std::string>
effective_dimension_names(const ClusteringEvaluationOptions &options,
                          std::int64_t dimensions) {
  if (!options.dimension_names.empty() &&
      options.dimension_names.size() != static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("clustering evaluation dimension_names must "
                                "have one name per semantic dimension");
  }

  std::vector<std::string> names;
  if (options.dimension_names.empty()) {
    names.reserve(static_cast<std::size_t>(dimensions));
    for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
      names.push_back("dimension_" + std::to_string(dimension));
    }
  } else {
    names = options.dimension_names;
  }

  std::set<std::string> unique_names;
  for (const auto &name : names) {
    if (name.empty()) {
      throw std::invalid_argument(
          "clustering evaluation dimension names must not be empty");
    }
    if (!unique_names.insert(name).second) {
      throw std::invalid_argument(
          "clustering evaluation dimension names must be unique");
    }
  }
  return names;
}

[[nodiscard]] ClusteringEvaluationOptions
effective_evaluation_options(const ClusteringEvaluationOptions &options,
                             std::int64_t dimensions) {
  switch (options.detail) {
  case ClusteringEvaluationDetail::Global:
  case ClusteringEvaluationDetail::Full:
    break;
  default:
    throw std::invalid_argument("clustering evaluation detail mode is invalid");
  }
  switch (options.semantic) {
  case SemanticEvaluationMode::Off:
  case SemanticEvaluationMode::Power:
    break;
  default:
    throw std::invalid_argument(
        "clustering evaluation semantic mode is invalid");
  }
  if (options.power != 1 && options.power != 2) {
    throw std::invalid_argument(
        "clustering evaluation semantic power must be 1 or 2");
  }

  const auto dimension_count = static_cast<std::size_t>(dimensions);
  if (!options.semantic_ranges.empty() &&
      options.semantic_ranges.size() != dimension_count) {
    throw std::invalid_argument("clustering evaluation semantic_ranges must "
                                "have one value per semantic dimension");
  }
  for (const auto range : options.semantic_ranges) {
    if (!std::isfinite(range) || !(range > 0.0)) {
      throw std::invalid_argument(
          "clustering evaluation semantic ranges must be finite and positive");
    }
  }

  if (!options.semantic_weights.empty() &&
      options.semantic_weights.size() != dimension_count) {
    throw std::invalid_argument("clustering evaluation semantic_weights must "
                                "have one value per semantic dimension");
  }
  bool has_positive_weight = false;
  for (const auto weight : options.semantic_weights) {
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("clustering evaluation semantic weights must "
                                  "be finite and non-negative");
    }
    has_positive_weight = has_positive_weight || weight > 0.0;
  }
  if (!options.semantic_weights.empty() && !has_positive_weight) {
    throw std::invalid_argument(
        "clustering evaluation requires at least one positive semantic weight");
  }

  ClusteringEvaluationOptions effective = options;
  effective.dimension_names = effective_dimension_names(options, dimensions);
  if (options.semantic == SemanticEvaluationMode::Power) {
    if (options.semantic_ranges.empty()) {
      throw std::invalid_argument(
          "clustering evaluation semantic ranges are required in power mode");
    }
    if (effective.semantic_weights.empty()) {
      effective.semantic_weights.assign(dimension_count, 1.0);
    }
  }
  return effective;
}

void require_uint64_values_fit_int64(const torch::Tensor &tensor,
                                     const char *what) {
  if (tensor.scalar_type() != c10::ScalarType::UInt64 || tensor.numel() == 0) {
    return;
  }
  auto cpu = tensor.detach().to(torch::kCPU).contiguous();
  const auto *values = cpu.data_ptr<std::uint64_t>();
  const auto limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  for (std::int64_t index = 0; index < cpu.numel(); ++index) {
    if (values[index] > limit) {
      throw std::invalid_argument(
          std::string("clustering evaluation ") + what +
          " uint64 values must be representable as int64");
    }
  }
}

} // namespace

const ClusteringMetricDescriptor &
clustering_metric_descriptor(ClusteringMetricId id) {
  const auto index = static_cast<std::size_t>(id);
  if (index >= kMetricDescriptors.size()) {
    throw std::invalid_argument("unknown clustering metric id");
  }
  return kMetricDescriptors[index];
}

std::span<const ClusteringMetricDescriptor> clustering_metric_descriptors() {
  return kMetricDescriptors;
}

std::span<const ClusteringMetricDescriptor>
exact_clustering_metric_descriptors() {
  return std::span<const ClusteringMetricDescriptor>(kMetricDescriptors.data(),
                                                     kExactMetricCount);
}

ClusteringEvaluationResult
evaluate_clustering(const torch::Tensor &predicted,
                    const torch::Tensor &semantic_truth,
                    const ClusteringEvaluationOptions &options) {
  if (!predicted.defined() || !semantic_truth.defined()) {
    throw std::invalid_argument(
        "clustering evaluation inputs must be defined tensors");
  }
  if (predicted.layout() != torch::kStrided ||
      semantic_truth.layout() != torch::kStrided) {
    throw std::invalid_argument(
        "clustering evaluation inputs must use strided tensor layouts");
  }
  if (!c10::isIntegralType(predicted.scalar_type(), /*includeBool=*/true)) {
    throw std::invalid_argument(
        "clustering evaluation predicted labels must have an integral dtype");
  }
  const auto truth_type = semantic_truth.scalar_type();
  const auto truth_is_floating = c10::isFloatingType(truth_type);
  if (!truth_is_floating &&
      !c10::isIntegralType(truth_type, /*includeBool=*/true)) {
    throw std::invalid_argument("clustering evaluation semantic truth must "
                                "have an integral or floating dtype");
  }
  require_uint64_values_fit_int64(predicted, "predicted labels");

  bool batched = false;
  if (predicted.dim() == 1 && semantic_truth.dim() == 2) {
    batched = false;
  } else if (predicted.dim() == 2 && semantic_truth.dim() == 3) {
    batched = true;
  } else {
    throw std::invalid_argument(
        "clustering evaluation expects predicted [N] with truth [N,D], or "
        "predicted [U,N] with truth [U,N,D]");
  }

  auto predicted_work =
      predicted.detach().to(torch::kLong).to(torch::kCPU).contiguous();
  auto truth_output = semantic_truth.detach().to(torch::kCPU).contiguous();
  if (!batched) {
    predicted_work = predicted_work.unsqueeze(0);
    truth_output = truth_output.unsqueeze(0);
  }

  const auto units = predicted_work.size(0);
  const auto observations = predicted_work.size(1);
  const auto dimensions = truth_output.size(2);
  if (units <= 0 || observations <= 0 || dimensions <= 0) {
    throw std::invalid_argument(
        "clustering evaluation inputs must have non-empty unit, observation, "
        "and dimension axes");
  }
  if (truth_output.size(0) != units || truth_output.size(1) != observations) {
    throw std::invalid_argument("clustering evaluation predicted and semantic "
                                "truth shapes do not align");
  }

  const auto truth_is_uint64 = truth_type == c10::ScalarType::UInt64;
  torch::Tensor truth_key_values;
  if (truth_is_floating) {
    if (!torch::isfinite(truth_output).all().item<bool>()) {
      throw std::invalid_argument("clustering evaluation semantic truth must "
                                  "contain only finite values");
    }
    truth_output = torch::where(truth_output == 0,
                                torch::zeros_like(truth_output), truth_output)
                       .contiguous();
    truth_key_values = truth_output.to(torch::kFloat64).contiguous();
  } else if (truth_is_uint64) {
    truth_key_values = truth_output;
  } else {
    truth_key_values = truth_output.to(torch::kLong).contiguous();
  }

  ClusteringEvaluationResult result;
  result.effective_options = effective_evaluation_options(options, dimensions);
  result.batched = batched;
  result.observation_count = observations;
  result.semantic_dimension_count = dimensions;
  result.units.reserve(static_cast<std::size_t>(units));
  for (std::int64_t unit = 0; unit < units; ++unit) {
    if (truth_is_floating) {
      result.units.push_back(evaluate_unit<double>(
          predicted_work[unit], truth_key_values[unit], truth_output[unit],
          result.effective_options, unit));
    } else if (truth_is_uint64) {
      result.units.push_back(evaluate_unit<std::uint64_t>(
          predicted_work[unit], truth_key_values[unit], truth_output[unit],
          result.effective_options, unit));
    } else {
      result.units.push_back(evaluate_unit<std::int64_t>(
          predicted_work[unit], truth_key_values[unit], truth_output[unit],
          result.effective_options, unit));
    }
  }
  return result;
}

} // namespace leakflow::ml
