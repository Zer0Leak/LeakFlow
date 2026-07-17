#include "leakflow/ml/clustering_evaluation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
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
constexpr std::array<ClusteringMetricDescriptor, 29> kMetricDescriptors{{
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
    {ClusteringMetricId::ExactAlignmentMatchedAccuracy,
     "exact_alignment_matched_accuracy", MetricFamily::Alignment,
     MetricDirection::HigherIsBetter, MetricAveraging::Micro},
    {ClusteringMetricId::ExactAlignmentPrecisionPerGroup,
     "exact_alignment_precision_per_group", MetricFamily::Alignment,
     MetricDirection::HigherIsBetter, MetricAveraging::PerGroup},
    {ClusteringMetricId::ExactAlignmentRecallPerGroup,
     "exact_alignment_recall_per_group", MetricFamily::Alignment,
     MetricDirection::HigherIsBetter, MetricAveraging::PerGroup},
    {ClusteringMetricId::ExactAlignmentF1PerGroup,
     "exact_alignment_f1_per_group", MetricFamily::Alignment,
     MetricDirection::HigherIsBetter, MetricAveraging::PerGroup},
    {ClusteringMetricId::ExactAlignmentJaccardPerGroup,
     "exact_alignment_jaccard_per_group", MetricFamily::Alignment,
     MetricDirection::HigherIsBetter, MetricAveraging::PerGroup},
    {ClusteringMetricId::SemanticAlignmentCost, "semantic_alignment_cost",
     MetricFamily::Alignment, MetricDirection::LowerIsBetter,
     MetricAveraging::Micro},
    {ClusteringMetricId::SemanticAlignmentDimensionError,
     "semantic_alignment_dimension_error", MetricFamily::Alignment,
     MetricDirection::LowerIsBetter, MetricAveraging::PerDimension},
    {ClusteringMetricId::SemanticPartitionSeparation,
     "semantic_partition_separation", MetricFamily::Semantic,
     MetricDirection::HigherIsBetter, MetricAveraging::Micro},
    {ClusteringMetricId::SemanticPartitionQuality, "semantic_partition_quality",
     MetricFamily::Combined, MetricDirection::HigherIsBetter,
     MetricAveraging::Micro},
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

template <typename Cost> struct AssignmentSolution {
  std::vector<std::size_t> column_for_row;
  std::vector<std::size_t> row_for_column;
  std::vector<Cost> row_dual;
  std::vector<Cost> column_dual;
};

template <typename Cost>
[[nodiscard]] AssignmentSolution<Cost>
hungarian_min_assignment(const std::vector<std::vector<Cost>> &cost) {
  const auto size = cost.size();
  if (size == 0) {
    throw std::invalid_argument(
        "clustering evaluation assignment matrix must not be empty");
  }
  for (const auto &row : cost) {
    if (row.size() != size) {
      throw std::invalid_argument(
          "clustering evaluation assignment matrix must be square");
    }
  }

  // One-based Kuhn-Munkres potentials and matching. Reached flags avoid a
  // finite sentinel whose arithmetic could overflow exact integer costs.
  std::vector<Cost> row_dual(size + 1, Cost{0});
  std::vector<Cost> column_dual(size + 1, Cost{0});
  std::vector<std::size_t> row_for_column(size + 1, 0);
  std::vector<std::size_t> predecessor(size + 1, 0);
  for (std::size_t next_row = 1; next_row <= size; ++next_row) {
    row_for_column[0] = next_row;
    std::size_t current_column = 0;
    std::vector<Cost> minimum(size + 1, Cost{0});
    std::vector<std::uint8_t> reached(size + 1, 0);
    std::vector<std::uint8_t> used(size + 1, 0);
    do {
      used[current_column] = 1;
      const auto current_row = row_for_column[current_column];
      bool delta_set = false;
      Cost delta{0};
      std::size_t next_column = 0;
      for (std::size_t column = 1; column <= size; ++column) {
        if (used[column] != 0) {
          continue;
        }
        const auto reduced = cost[current_row - 1][column - 1] -
                             row_dual[current_row] - column_dual[column];
        if (reached[column] == 0 || reduced < minimum[column]) {
          minimum[column] = reduced;
          reached[column] = 1;
          predecessor[column] = current_column;
        }
        if (!delta_set || minimum[column] < delta) {
          delta = minimum[column];
          next_column = column;
          delta_set = true;
        }
      }
      if (!delta_set) {
        throw std::logic_error(
            "clustering evaluation assignment lost an augmenting path");
      }
      for (std::size_t column = 0; column <= size; ++column) {
        if (used[column] != 0) {
          row_dual[row_for_column[column]] += delta;
          column_dual[column] -= delta;
        } else if (reached[column] != 0) {
          minimum[column] -= delta;
        }
      }
      current_column = next_column;
    } while (row_for_column[current_column] != 0);

    do {
      const auto previous_column = predecessor[current_column];
      row_for_column[current_column] = row_for_column[previous_column];
      current_column = previous_column;
    } while (current_column != 0);
  }

  AssignmentSolution<Cost> result;
  result.column_for_row.resize(size);
  result.row_for_column.resize(size);
  result.row_dual.resize(size);
  result.column_dual.resize(size);
  for (std::size_t column = 1; column <= size; ++column) {
    const auto row = row_for_column[column] - 1;
    result.column_for_row[row] = column - 1;
    result.row_for_column[column - 1] = row;
  }
  for (std::size_t index = 0; index < size; ++index) {
    result.row_dual[index] = row_dual[index + 1];
    result.column_dual[index] = column_dual[index + 1];
  }
  return result;
}

template <typename Cost>
void refine_assignment_lexicographically(
    const std::vector<std::vector<Cost>> &cost,
    AssignmentSolution<Cost> &solution) {
  const auto size = cost.size();
  const auto no_index = std::numeric_limits<std::size_t>::max();

  // Every primary-optimal matching uses only zero-reduced-cost edges. Keep the
  // initial matching explicitly because recomputing a selected long-double
  // edge can leave a tiny residual.
  std::vector<std::vector<std::uint8_t>> equality(
      size, std::vector<std::uint8_t>(size, 0));
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column < size; ++column) {
      const auto reduced = cost[row][column] - solution.row_dual[row] -
                           solution.column_dual[column];
      equality[row][column] = static_cast<std::uint8_t>(
          solution.column_for_row[row] == column || reduced == Cost{0});
    }
  }

  std::vector<std::uint8_t> fixed_column(size, 0);
  for (std::size_t row = 0; row < size; ++row) {
    const auto current_column = solution.column_for_row[row];
    std::vector<std::uint8_t> reachable_row(size, 0);
    std::vector<std::uint8_t> reachable_column(size, 0);
    std::vector<std::size_t> next_column_for_row(size, no_index);
    std::deque<std::size_t> queue;
    reachable_column[current_column] = 1;
    queue.push_back(current_column);

    // Reverse alternating reachability from the column made free by removing
    // this row. Earlier rows are already fixed and excluded.
    while (!queue.empty()) {
      const auto column = queue.front();
      queue.pop_front();
      for (std::size_t candidate_row = row + 1; candidate_row < size;
           ++candidate_row) {
        if (reachable_row[candidate_row] != 0 ||
            solution.column_for_row[candidate_row] == column ||
            equality[candidate_row][column] == 0) {
          continue;
        }
        reachable_row[candidate_row] = 1;
        next_column_for_row[candidate_row] = column;
        const auto matched_column = solution.column_for_row[candidate_row];
        if (reachable_column[matched_column] == 0) {
          reachable_column[matched_column] = 1;
          queue.push_back(matched_column);
        }
      }
    }

    std::size_t chosen_column = no_index;
    for (std::size_t candidate_column = 0; candidate_column < size;
         ++candidate_column) {
      if (fixed_column[candidate_column] != 0 ||
          equality[row][candidate_column] == 0) {
        continue;
      }
      if (candidate_column == current_column) {
        chosen_column = candidate_column;
        break;
      }
      const auto displaced_row = solution.row_for_column[candidate_column];
      if (displaced_row > row && reachable_row[displaced_row] != 0) {
        chosen_column = candidate_column;
        break;
      }
    }
    if (chosen_column == no_index) {
      throw std::logic_error(
          "clustering evaluation lexicographic assignment lost a matching");
    }

    if (chosen_column != current_column) {
      auto displaced_row = solution.row_for_column[chosen_column];
      solution.column_for_row[row] = chosen_column;
      solution.row_for_column[chosen_column] = row;
      std::size_t steps = 0;
      while (true) {
        if (++steps > size) {
          throw std::logic_error(
              "clustering evaluation assignment refinement found a cycle");
        }
        const auto next_column = next_column_for_row[displaced_row];
        if (next_column == no_index) {
          throw std::logic_error(
              "clustering evaluation assignment refinement lost its path");
        }
        const auto next_displaced_row = solution.row_for_column[next_column];
        solution.column_for_row[displaced_row] = next_column;
        solution.row_for_column[next_column] = displaced_row;
        if (next_column == current_column) {
          break;
        }
        displaced_row = next_displaced_row;
      }
    }
    fixed_column[chosen_column] = 1;
  }
}

template <typename Cost>
[[nodiscard]] AssignmentSolution<Cost>
lexicographic_min_assignment(const std::vector<std::vector<Cost>> &cost) {
  auto result = hungarian_min_assignment(cost);
  refine_assignment_lexicographically(cost, result);
  return result;
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

[[nodiscard]] std::uint64_t
contingency_count(const ContingencyCounts &contingency,
                  std::int64_t truth_group, std::int64_t predicted_cluster) {
  const auto found = contingency.find({truth_group, predicted_cluster});
  return found == contingency.end() ? 0 : found->second;
}

template <typename Cost>
[[nodiscard]] ClusteringAlignmentMapping
build_alignment_mapping(const AssignmentSolution<Cost> &assignment,
                        const std::vector<std::uint64_t> &truth_marginal,
                        const std::vector<std::uint64_t> &predicted_marginal) {
  const auto truth_groups = truth_marginal.size();
  const auto predicted_clusters = predicted_marginal.size();
  ClusteringAlignmentMapping result;
  result.predicted_to_truth_group.assign(predicted_clusters,
                                         kUnmatchedAlignmentIndex);
  result.truth_to_predicted_cluster.assign(truth_groups,
                                           kUnmatchedAlignmentIndex);

  for (std::size_t predicted = 0; predicted < predicted_clusters; ++predicted) {
    const auto column = assignment.column_for_row[predicted];
    if (column < truth_groups) {
      result.predicted_to_truth_group[predicted] =
          static_cast<std::int64_t>(column);
      result.truth_to_predicted_cluster[column] =
          static_cast<std::int64_t>(predicted);
      result.assigned_predicted_observation_count =
          checked_add(result.assigned_predicted_observation_count,
                      predicted_marginal[predicted],
                      "assigned predicted alignment support");
    } else {
      result.unmatched_predicted_clusters.push_back(
          {static_cast<std::int64_t>(predicted),
           predicted_marginal[predicted]});
      result.unmatched_predicted_observation_count =
          checked_add(result.unmatched_predicted_observation_count,
                      predicted_marginal[predicted],
                      "unmatched predicted alignment support");
    }
  }

  for (std::size_t truth_group = 0; truth_group < truth_groups; ++truth_group) {
    if (result.truth_to_predicted_cluster[truth_group] !=
        kUnmatchedAlignmentIndex) {
      result.assigned_truth_observation_count = checked_add(
          result.assigned_truth_observation_count, truth_marginal[truth_group],
          "assigned truth alignment support");
    } else {
      result.unmatched_truth_groups.push_back(
          {static_cast<std::int64_t>(truth_group),
           truth_marginal[truth_group]});
      result.unmatched_truth_observation_count = checked_add(
          result.unmatched_truth_observation_count, truth_marginal[truth_group],
          "unmatched truth alignment support");
    }
  }
  return result;
}

[[nodiscard]] ExactOverlapAlignment compute_exact_overlap_alignment(
    const ContingencyCounts &contingency,
    const std::vector<std::uint64_t> &truth_marginal,
    const std::vector<std::uint64_t> &predicted_marginal,
    std::uint64_t observations, ClusteringEvaluationDetail detail) {
  const auto truth_groups = truth_marginal.size();
  const auto predicted_clusters = predicted_marginal.size();
  const auto assignment_size = std::max(truth_groups, predicted_clusters);
  using ExactCost = __int128;
  std::vector<std::vector<ExactCost>> cost(
      assignment_size, std::vector<ExactCost>(assignment_size, 0));
  for (const auto &[cell, count] : contingency) {
    const auto truth_group = static_cast<std::size_t>(cell.first);
    const auto predicted_cluster = static_cast<std::size_t>(cell.second);
    cost[predicted_cluster][truth_group] = -static_cast<ExactCost>(count);
  }

  const auto assignment = lexicographic_min_assignment(cost);
  ExactOverlapAlignment result;
  result.mapping =
      build_alignment_mapping(assignment, truth_marginal, predicted_marginal);
  for (std::size_t predicted = 0; predicted < predicted_clusters; ++predicted) {
    const auto truth_group = result.mapping.predicted_to_truth_group[predicted];
    if (truth_group == kUnmatchedAlignmentIndex) {
      continue;
    }
    result.matched_overlap_observation_count =
        checked_add(result.matched_overlap_observation_count,
                    contingency_count(contingency, truth_group,
                                      static_cast<std::int64_t>(predicted)),
                    "exact alignment matched observations");
  }
  if (result.matched_overlap_observation_count > observations) {
    throw std::logic_error(
        "clustering evaluation exact alignment support is inconsistent");
  }
  result.nonmatched_observation_count =
      observations - result.matched_overlap_observation_count;
  result.matched_accuracy = defined_metric(
      ClusteringMetricId::ExactAlignmentMatchedAccuracy,
      static_cast<double>(result.matched_overlap_observation_count) /
          static_cast<double>(observations),
      observations);

  result.aligned_column_to_predicted_cluster.reserve(predicted_clusters);
  for (const auto predicted : result.mapping.truth_to_predicted_cluster) {
    if (predicted != kUnmatchedAlignmentIndex) {
      result.aligned_column_to_predicted_cluster.push_back(predicted);
    }
  }
  for (const auto &unmatched : result.mapping.unmatched_predicted_clusters) {
    result.aligned_column_to_predicted_cluster.push_back(unmatched.index);
  }
  if (result.aligned_column_to_predicted_cluster.size() != predicted_clusters) {
    throw std::logic_error(
        "clustering evaluation exact alignment permutation is inconsistent");
  }

  if (detail == ClusteringEvaluationDetail::Full) {
    result.truth_group_details.emplace();
    result.truth_group_details->reserve(truth_groups);
    for (std::size_t truth_group = 0; truth_group < truth_groups;
         ++truth_group) {
      ExactAlignmentTruthGroupDetail group;
      group.truth_group_index = static_cast<std::int64_t>(truth_group);
      group.predicted_cluster_index =
          result.mapping.truth_to_predicted_cluster[truth_group];
      group.truth_observation_count = truth_marginal[truth_group];

      if (group.predicted_cluster_index == kUnmatchedAlignmentIndex) {
        group.precision = undefined_metric(
            ClusteringMetricId::ExactAlignmentPrecisionPerGroup, 0,
            MetricUndefinedReason::NoMatchedPredictedCluster);
        group.recall =
            defined_metric(ClusteringMetricId::ExactAlignmentRecallPerGroup,
                           0.0, group.truth_observation_count);
        group.f1 =
            undefined_metric(ClusteringMetricId::ExactAlignmentF1PerGroup,
                             group.truth_observation_count,
                             MetricUndefinedReason::DependentMetricUndefined);
        group.jaccard =
            defined_metric(ClusteringMetricId::ExactAlignmentJaccardPerGroup,
                           0.0, group.truth_observation_count);
      } else {
        const auto predicted =
            static_cast<std::size_t>(group.predicted_cluster_index);
        group.predicted_observation_count = predicted_marginal[predicted];
        group.overlap_observation_count = contingency_count(
            contingency, static_cast<std::int64_t>(truth_group),
            group.predicted_cluster_index);
        group.precision = defined_metric(
            ClusteringMetricId::ExactAlignmentPrecisionPerGroup,
            static_cast<double>(group.overlap_observation_count) /
                static_cast<double>(group.predicted_observation_count),
            group.predicted_observation_count);
        group.recall = defined_metric(
            ClusteringMetricId::ExactAlignmentRecallPerGroup,
            static_cast<double>(group.overlap_observation_count) /
                static_cast<double>(group.truth_observation_count),
            group.truth_observation_count);
        const auto f1_support = checked_add(group.truth_observation_count,
                                            group.predicted_observation_count,
                                            "exact alignment F1 support");
        group.f1 = defined_metric(
            ClusteringMetricId::ExactAlignmentF1PerGroup,
            2.0 * static_cast<double>(group.overlap_observation_count) /
                static_cast<double>(f1_support),
            f1_support);
        if (group.overlap_observation_count > f1_support) {
          throw std::logic_error(
              "clustering evaluation exact alignment overlap is inconsistent");
        }
        const auto jaccard_support =
            f1_support - group.overlap_observation_count;
        group.jaccard = defined_metric(
            ClusteringMetricId::ExactAlignmentJaccardPerGroup,
            static_cast<double>(group.overlap_observation_count) /
                static_cast<double>(jaccard_support),
            jaccard_support);
      }
      result.truth_group_details->push_back(std::move(group));
    }
  }
  return result;
}

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
  long double aggregate_all = 0.0L;
  std::vector<long double> dimension_all;
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

[[nodiscard]] long double validated_bounded_sum(long double value,
                                                std::uint64_t upper_count,
                                                const char *what) {
  const auto upper = static_cast<long double>(upper_count);
  const auto tolerance = 64.0L * std::numeric_limits<long double>::epsilon() *
                         std::max(1.0L, upper);
  if (value < -tolerance || value > upper + tolerance ||
      !std::isfinite(value)) {
    throw std::runtime_error(std::string("clustering evaluation ") + what +
                             " aggregation failed");
  }
  return std::clamp(value, 0.0L, upper);
}

[[nodiscard]] long double validated_pair_sum(long double value,
                                             std::uint64_t pair_count) {
  return validated_bounded_sum(value, pair_count, "semantic pair");
}

[[nodiscard]] long double
absolute_pair_sum(std::vector<long double> &coordinate_values) {
  std::sort(coordinate_values.begin(), coordinate_values.end());
  CompensatedSum prefix;
  CompensatedSum pair_sum;
  for (std::size_t index = 0; index < coordinate_values.size(); ++index) {
    const auto value = coordinate_values[index];
    pair_sum.add(static_cast<long double>(index) * value - prefix.value());
    prefix.add(value);
  }
  return pair_sum.value();
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
  result.dimension_all.assign(dimension_count, 0.0L);
  const auto all_pair_count =
      checked_comb2(static_cast<std::uint64_t>(observations));

  if (options.power == 2) {
    std::vector<RunningVariance> moments(clusters * dimension_count);
    std::vector<RunningVariance> all_moments(dimension_count);
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
        all_moments[dimension_index].add(value);
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
    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
      result.dimension_all[dimension] =
          validated_pair_sum(all_moments[dimension].pair_sum(), all_pair_count);
    }
  } else {
    std::vector<std::vector<long double>> values(clusters * dimension_count);
    std::vector<std::vector<long double>> all_values(dimension_count);
    for (std::int64_t observation = 0; observation < observations;
         ++observation) {
      const auto cluster =
          static_cast<std::size_t>(predicted_assignment[observation]);
      for (std::int64_t dimension = 0; dimension < dimensions; ++dimension) {
        const auto dimension_index = static_cast<std::size_t>(dimension);
        const auto value =
            nonnegative_coordinate_difference(truth[observation][dimension],
                                              minima[dimension_index]) /
            static_cast<long double>(options.semantic_ranges[dimension_index]);
        values[cluster * dimension_count + dimension_index].push_back(value);
        all_values[dimension_index].push_back(value);
      }
    }
    for (std::size_t cluster = 0; cluster < clusters; ++cluster) {
      const auto pair_count = checked_comb2(predicted_marginal[cluster]);
      for (std::size_t dimension = 0; dimension < dimension_count;
           ++dimension) {
        auto &coordinate_values = values[cluster * dimension_count + dimension];
        result.dimension_by_cluster[cluster][dimension] = validated_pair_sum(
            absolute_pair_sum(coordinate_values), pair_count);
      }
    }
    for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
      result.dimension_all[dimension] = validated_pair_sum(
          absolute_pair_sum(all_values[dimension]), all_pair_count);
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
  CompensatedSum all_weighted_sum;
  for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
    all_weighted_sum.add(static_cast<long double>(
                             options.semantic_weights[dimension] / max_weight) *
                         result.dimension_all[dimension]);
  }
  result.aggregate_all = validated_pair_sum(
      all_weighted_sum.value() / scaled_weight_sum.value(), all_pair_count);
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
  result.partition_separation = undefined_metric(
      ClusteringMetricId::SemanticPartitionSeparation, pairs.total_pairs,
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
  const auto semantic_total =
      validated_pair_sum(pair_sums.aggregate_all, pairs.total_pairs);
  std::vector<long double> dimension_error_totals(dimension_count, 0.0L);
  for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
    dimension_error_totals[dimension] = validated_pair_sum(
        dimension_totals[dimension].value(), pairs.false_positive);
  }

  if (semantic_total == 0.0L) {
    result.partition_separation = undefined_metric(
        ClusteringMetricId::SemanticPartitionSeparation, pairs.total_pairs,
        MetricUndefinedReason::NoSemanticVariation);
  } else {
    const auto separation = 1.0L - semantic_error_total / semantic_total;
    result.partition_separation =
        defined_metric(ClusteringMetricId::SemanticPartitionSeparation,
                       static_cast<double>(validated_bounded_sum(
                           separation, 1, "semantic separation")),
                       pairs.total_pairs);
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

[[nodiscard]] SemanticPartitionClusteringQuality
compute_semantic_partition_quality(const SemanticClusteringMetrics &semantic,
                                   const ExactClusteringMetrics &exact,
                                   std::uint64_t observations) {
  SemanticPartitionClusteringQuality result;
  result.semantic_partition_separation = semantic.partition_separation;
  result.pair_recall = exact.pair_recall;
  if (!semantic.partition_separation.defined() ||
      !exact.pair_recall.defined()) {
    result.quality = undefined_metric(
        ClusteringMetricId::SemanticPartitionQuality, observations,
        MetricUndefinedReason::DependentMetricUndefined);
    return result;
  }

  const auto separation = validated_bounded_sum(
      static_cast<long double>(*semantic.partition_separation.value), 1,
      "semantic partition separation");
  const auto pair_recall =
      validated_bounded_sum(static_cast<long double>(*exact.pair_recall.value),
                            1, "semantic partition pair recall");
  const auto denominator = separation + pair_recall;
  const auto quality = denominator == 0.0L
                           ? 0.0L
                           : (2.0L * separation * pair_recall) / denominator;
  result.quality =
      defined_metric(ClusteringMetricId::SemanticPartitionQuality,
                     static_cast<double>(validated_bounded_sum(
                         quality, 1, "semantic partition quality")),
                     observations);
  return result;
}

struct SemanticAlignmentWeightState {
  std::vector<long double> scaled_weights;
  long double scaled_weight_sum = 0.0L;
};

[[nodiscard]] SemanticAlignmentWeightState
semantic_alignment_weight_state(const ClusteringEvaluationOptions &options) {
  const auto maximum = *std::max_element(options.semantic_weights.begin(),
                                         options.semantic_weights.end());
  SemanticAlignmentWeightState result;
  result.scaled_weights.reserve(options.semantic_weights.size());
  CompensatedSum sum;
  for (const auto weight : options.semantic_weights) {
    const auto scaled = static_cast<long double>(weight / maximum);
    result.scaled_weights.push_back(scaled);
    sum.add(scaled);
  }
  result.scaled_weight_sum = sum.value();
  if (!(result.scaled_weight_sum > 0.0L) ||
      !std::isfinite(result.scaled_weight_sum)) {
    throw std::logic_error(
        "clustering evaluation semantic alignment weights are invalid");
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] long double
semantic_alignment_dimension_cost(Scalar left, Scalar right, double range,
                                  std::int64_t power) {
  const auto high = std::max(left, right);
  const auto low = std::min(left, right);
  const auto normalized = nonnegative_coordinate_difference(high, low) /
                          static_cast<long double>(range);
  const auto cost = power == 2 ? normalized * normalized : normalized;
  return validated_bounded_sum(cost, 1, "semantic alignment dimension");
}

[[nodiscard]] long double semantic_alignment_aggregate_cost(
    const std::vector<long double> &dimension_costs,
    const SemanticAlignmentWeightState &weights) {
  CompensatedSum weighted;
  for (std::size_t dimension = 0; dimension < dimension_costs.size();
       ++dimension) {
    weighted.add(weights.scaled_weights[dimension] *
                 dimension_costs[dimension]);
  }
  return validated_bounded_sum(weighted.value() / weights.scaled_weight_sum, 1,
                               "semantic alignment cost");
}

template <typename Scalar>
[[nodiscard]] SemanticCostAlignment compute_semantic_cost_alignment(
    const std::vector<std::vector<Scalar>> &truth_vectors,
    const ContingencyCounts &contingency,
    const std::vector<std::uint64_t> &truth_marginal,
    const std::vector<std::uint64_t> &predicted_marginal,
    const ClusteringEvaluationOptions &options, std::uint64_t observations) {
  const auto truth_groups = truth_vectors.size();
  const auto predicted_clusters = predicted_marginal.size();
  const auto dimensions = truth_vectors.front().size();
  const auto assignment_size = std::max(truth_groups, predicted_clusters);
  const auto weights = semantic_alignment_weight_state(options);

  // Truth-to-truth aggregate costs are symmetric and independent of predicted
  // cluster membership. The assignment matrix then uses only sparse
  // contingency cells, never observation pairs.
  std::vector<std::vector<long double>> truth_cost(
      truth_groups, std::vector<long double>(truth_groups, 0.0L));
  std::vector<long double> dimension_costs(dimensions, 0.0L);
  for (std::size_t left = 0; left < truth_groups; ++left) {
    for (std::size_t right = left + 1; right < truth_groups; ++right) {
      for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        dimension_costs[dimension] = semantic_alignment_dimension_cost(
            truth_vectors[left][dimension], truth_vectors[right][dimension],
            options.semantic_ranges[dimension], options.power);
      }
      const auto aggregate =
          semantic_alignment_aggregate_cost(dimension_costs, weights);
      truth_cost[left][right] = aggregate;
      truth_cost[right][left] = aggregate;
    }
  }

  std::vector<std::vector<std::pair<std::size_t, std::uint64_t>>>
      cells_by_predicted(predicted_clusters);
  for (const auto &[cell, count] : contingency) {
    cells_by_predicted[static_cast<std::size_t>(cell.second)].push_back(
        {static_cast<std::size_t>(cell.first), count});
  }

  std::vector<std::vector<long double>> cost(
      assignment_size, std::vector<long double>(assignment_size, 0.0L));
  for (std::size_t predicted = 0; predicted < predicted_clusters; ++predicted) {
    for (std::size_t target_truth = 0; target_truth < truth_groups;
         ++target_truth) {
      CompensatedSum total;
      for (const auto &[source_truth, count] : cells_by_predicted[predicted]) {
        total.add(static_cast<long double>(count) *
                  truth_cost[source_truth][target_truth]);
      }
      cost[predicted][target_truth] =
          validated_bounded_sum(total.value(), predicted_marginal[predicted],
                                "semantic alignment matrix");
    }
    for (std::size_t dummy_truth = truth_groups; dummy_truth < assignment_size;
         ++dummy_truth) {
      cost[predicted][dummy_truth] =
          static_cast<long double>(predicted_marginal[predicted]);
    }
  }

  const auto assignment = lexicographic_min_assignment(cost);
  SemanticCostAlignment result;
  result.mapping =
      build_alignment_mapping(assignment, truth_marginal, predicted_marginal);
  if (options.detail == ClusteringEvaluationDetail::Full) {
    result.error_masses.emplace();
    result.error_masses->reserve(contingency.size());
  }

  std::vector<CompensatedSum> dimension_totals(dimensions);
  CompensatedSum aggregate_total;
  for (const auto &[cell, count] : contingency) {
    const auto source_truth = static_cast<std::size_t>(cell.first);
    const auto predicted = static_cast<std::size_t>(cell.second);
    const auto assigned_truth =
        result.mapping.predicted_to_truth_group[predicted];
    std::vector<double> output_dimension_costs;
    if (result.error_masses.has_value()) {
      output_dimension_costs.reserve(dimensions);
    }
    if (assigned_truth == kUnmatchedAlignmentIndex) {
      std::fill(dimension_costs.begin(), dimension_costs.end(), 1.0L);
    } else {
      const auto target_truth = static_cast<std::size_t>(assigned_truth);
      for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        dimension_costs[dimension] = semantic_alignment_dimension_cost(
            truth_vectors[source_truth][dimension],
            truth_vectors[target_truth][dimension],
            options.semantic_ranges[dimension], options.power);
      }
    }
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
      dimension_totals[dimension].add(static_cast<long double>(count) *
                                      dimension_costs[dimension]);
      if (result.error_masses.has_value()) {
        output_dimension_costs.push_back(
            static_cast<double>(dimension_costs[dimension]));
      }
    }
    const auto aggregate =
        assigned_truth == kUnmatchedAlignmentIndex
            ? 1.0L
            : semantic_alignment_aggregate_cost(dimension_costs, weights);
    aggregate_total.add(static_cast<long double>(count) * aggregate);
    if (assigned_truth == cell.first) {
      result.exact_overlap_observation_count =
          checked_add(result.exact_overlap_observation_count, count,
                      "semantic alignment exact overlap");
    }
    if (result.error_masses.has_value()) {
      result.error_masses->push_back({cell.first, cell.second, assigned_truth,
                                      count, static_cast<double>(aggregate),
                                      std::move(output_dimension_costs)});
    }
  }

  const auto bounded_total = validated_bounded_sum(
      aggregate_total.value(), observations, "semantic alignment total");
  result.normalized_cost = defined_metric(
      ClusteringMetricId::SemanticAlignmentCost,
      static_cast<double>(bounded_total /
                          static_cast<long double>(observations)),
      observations);
  result.dimensions.reserve(dimensions);
  for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
    const auto bounded_dimension =
        validated_bounded_sum(dimension_totals[dimension].value(), observations,
                              "semantic alignment dimension total");
    result.dimensions.push_back(
        {static_cast<std::int64_t>(dimension),
         defined_metric(
             ClusteringMetricId::SemanticAlignmentDimensionError,
             static_cast<double>(bounded_dimension /
                                 static_cast<long double>(observations)),
             observations)});
  }
  if (result.exact_overlap_observation_count > observations) {
    throw std::logic_error(
        "clustering evaluation semantic alignment overlap is inconsistent");
  }
  result.nonoverlap_observation_count =
      observations - result.exact_overlap_observation_count;
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
  std::vector<std::vector<Scalar>> truth_vectors;
  truth_representative_rows.reserve(unique_truth.size());
  truth_vectors.reserve(unique_truth.size());
  for (const auto &[key, representative_row] : unique_truth) {
    const auto index =
        static_cast<std::int64_t>(truth_representative_rows.size());
    truth_index.emplace(key, index);
    truth_representative_rows.push_back(representative_row);
    truth_vectors.push_back(key);
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

  const auto observations_u64 = static_cast<std::uint64_t>(observations);
  if (options.semantic_partition_quality) {
    result.semantic_partition_quality = compute_semantic_partition_quality(
        result.semantic, result.exact, observations_u64);
  }
  const auto exact_alignment_requested =
      options.alignment == AlignmentEvaluationMode::Exact ||
      options.alignment == AlignmentEvaluationMode::Both;
  const auto semantic_alignment_requested =
      options.alignment == AlignmentEvaluationMode::Semantic ||
      options.alignment == AlignmentEvaluationMode::Both;
  if (exact_alignment_requested) {
    result.exact_alignment = compute_exact_overlap_alignment(
        contingency, truth_marginal, predicted_marginal, observations_u64,
        options.detail);
  }
  if (semantic_alignment_requested) {
    result.semantic_alignment = compute_semantic_cost_alignment(
        truth_vectors, contingency, truth_marginal, predicted_marginal, options,
        observations_u64);
  }

  torch::Tensor canonical_predicted_ids;
  torch::Tensor canonical_truth_vectors;
  const auto alignment_requested =
      options.alignment != AlignmentEvaluationMode::None;
  if (alignment_requested ||
      options.detail == ClusteringEvaluationDetail::Full) {
    canonical_predicted_ids = torch::tensor(predicted_ids, torch::kLong);
    const auto representative_index =
        torch::tensor(truth_representative_rows, torch::kLong);
    canonical_truth_vectors =
        truth_output_values.index_select(0, representative_index).contiguous();
  }
  if (alignment_requested) {
    result.alignment_identities = ClusteringAlignmentIdentities{
        canonical_predicted_ids, canonical_truth_vectors};
  }

  if (options.detail == ClusteringEvaluationDetail::Full) {
    ClusteringPartitionDetail partition;
    partition.predicted_ids = canonical_predicted_ids;
    partition.truth_vectors = canonical_truth_vectors;

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
  switch (options.alignment) {
  case AlignmentEvaluationMode::None:
  case AlignmentEvaluationMode::Exact:
  case AlignmentEvaluationMode::Semantic:
  case AlignmentEvaluationMode::Both:
    break;
  default:
    throw std::invalid_argument(
        "clustering evaluation alignment mode is invalid");
  }
  if ((options.alignment == AlignmentEvaluationMode::Semantic ||
       options.alignment == AlignmentEvaluationMode::Both) &&
      options.semantic != SemanticEvaluationMode::Power) {
    throw std::invalid_argument("clustering evaluation semantic alignment "
                                "requires power semantic evaluation");
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
  effective.semantic_partition_quality =
      options.semantic == SemanticEvaluationMode::Power &&
      options.semantic_partition_quality;
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

std::string_view metric_family_name(MetricFamily family) {
  switch (family) {
  case MetricFamily::Exact:
    return "exact";
  case MetricFamily::Semantic:
    return "semantic";
  case MetricFamily::Fragmentation:
    return "fragmentation";
  case MetricFamily::Alignment:
    return "alignment";
  case MetricFamily::Combined:
    return "combined";
  }
  throw std::invalid_argument("unknown clustering metric family");
}

std::string_view metric_direction_name(MetricDirection direction) {
  switch (direction) {
  case MetricDirection::HigherIsBetter:
    return "higher_is_better";
  case MetricDirection::LowerIsBetter:
    return "lower_is_better";
  }
  throw std::invalid_argument("unknown clustering metric direction");
}

std::string_view metric_averaging_name(MetricAveraging averaging) {
  switch (averaging) {
  case MetricAveraging::None:
    return "none";
  case MetricAveraging::Micro:
    return "micro";
  case MetricAveraging::Macro:
    return "macro";
  case MetricAveraging::PerCluster:
    return "per_cluster";
  case MetricAveraging::PerGroup:
    return "per_group";
  case MetricAveraging::PerDimension:
    return "per_dimension";
  }
  throw std::invalid_argument("unknown clustering metric averaging");
}

std::string_view metric_undefined_reason_name(MetricUndefinedReason reason) {
  switch (reason) {
  case MetricUndefinedReason::None:
    return "none";
  case MetricUndefinedReason::NoPredictedWithinClusterPairs:
    return "no_predicted_within_cluster_pairs";
  case MetricUndefinedReason::NoTrueWithinGroupPairs:
    return "no_true_within_group_pairs";
  case MetricUndefinedReason::DependentMetricUndefined:
    return "dependent_metric_undefined";
  case MetricUndefinedReason::SemanticDisabled:
    return "semantic_disabled";
  case MetricUndefinedReason::NoEligiblePredictedClusters:
    return "no_eligible_predicted_clusters";
  case MetricUndefinedReason::NoMergeErrorPairs:
    return "no_merge_error_pairs";
  case MetricUndefinedReason::NoEligibleTruthGroups:
    return "no_eligible_truth_groups";
  case MetricUndefinedReason::NoMatchedPredictedCluster:
    return "no_matched_predicted_cluster";
  case MetricUndefinedReason::NoSemanticVariation:
    return "no_semantic_variation";
  }
  throw std::invalid_argument("unknown clustering metric undefined reason");
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
