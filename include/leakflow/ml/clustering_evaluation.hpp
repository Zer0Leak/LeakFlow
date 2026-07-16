#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <vector>

namespace leakflow::ml {

enum class ClusteringMetricId : std::uint8_t {
  AdjustedRandIndex,
  AdjustedMutualInformation,
  Homogeneity,
  Completeness,
  VMeasure,
  Purity,
  PairPrecision,
  PairRecall,
  PairF1,
  NormalizedMutualInformation,
};

enum class MetricFamily : std::uint8_t {
  Exact,
  Semantic,
  Fragmentation,
  Alignment,
};

enum class MetricDirection : std::uint8_t {
  HigherIsBetter,
  LowerIsBetter,
};

enum class MetricAveraging : std::uint8_t {
  None,
  Micro,
  Macro,
  PerCluster,
  PerGroup,
  PerDimension,
};

struct ClusteringMetricDescriptor {
  ClusteringMetricId id;
  std::string_view name;
  MetricFamily family;
  MetricDirection direction;
  MetricAveraging averaging;
};

// Descriptors are stable, process-lifetime data. The exact descriptor span uses
// the same order as ClusteringMetricId.
[[nodiscard]] const ClusteringMetricDescriptor &
clustering_metric_descriptor(ClusteringMetricId id);
[[nodiscard]] std::span<const ClusteringMetricDescriptor>
exact_clustering_metric_descriptors();

enum class MetricUndefinedReason : std::uint8_t {
  None,
  NoPredictedWithinClusterPairs,
  NoTrueWithinGroupPairs,
  DependentMetricUndefined,
};

struct MetricValue {
  ClusteringMetricId metric = ClusteringMetricId::AdjustedRandIndex;
  std::optional<double> value;
  std::uint64_t support_count = 0;
  MetricUndefinedReason undefined_reason = MetricUndefinedReason::None;

  [[nodiscard]] bool defined() const noexcept { return value.has_value(); }
};

struct PairCounts {
  // Counts use unordered observation pairs (i < j). This is half the raw
  // count convention used by sklearn.metrics.cluster.pair_confusion_matrix;
  // pair-score ratios are unchanged.
  std::uint64_t total_pairs = 0;
  std::uint64_t true_positive = 0;
  std::uint64_t false_positive = 0;
  std::uint64_t false_negative = 0;
  std::uint64_t true_negative = 0;
  std::uint64_t predicted_within_cluster_pairs = 0; // TP + FP
  std::uint64_t true_within_group_pairs = 0;        // TP + FN
};

struct ExactClusteringMetrics {
  MetricValue adjusted_rand_index;
  MetricValue adjusted_mutual_information;
  MetricValue homogeneity;
  MetricValue completeness;
  MetricValue v_measure;
  MetricValue purity;
  MetricValue pair_precision;
  MetricValue pair_recall;
  MetricValue pair_f1;
  MetricValue normalized_mutual_information;
};

struct SparseContingency {
  // CPU, contiguous, int64 COO arrays sorted by (truth_group,
  // predicted_cluster).
  torch::Tensor truth_group_indices;       // [Z]
  torch::Tensor predicted_cluster_indices; // [Z]
  torch::Tensor counts;                    // [Z]
  std::int64_t truth_group_count = 0;
  std::int64_t predicted_cluster_count = 0;
};

struct ClusteringPartitionDetail {
  // Deterministic normalization: predicted ids are ascending and semantic
  // vectors are lexicographically ordered after lossless canonicalization.
  torch::Tensor predicted_ids; // [K], CPU int64
  torch::Tensor truth_vectors; // [G,D], CPU, original numeric dtype
  SparseContingency contingency;
};

struct ClusteringEvaluationUnitResult {
  std::int64_t observation_count = 0;
  std::int64_t truth_group_count = 0;
  std::int64_t predicted_cluster_count = 0;
  PairCounts pair_counts;
  ExactClusteringMetrics exact;
  std::optional<ClusteringPartitionDetail> partition_detail;
};

enum class ClusteringEvaluationDetail : std::uint8_t {
  Global,
  Full,
};

struct ClusteringEvaluationOptions {
  ClusteringEvaluationDetail detail = ClusteringEvaluationDetail::Global;
  std::vector<std::string> dimension_names;
};

struct ClusteringEvaluationResult {
  std::uint32_t schema_version = 1;
  ClusteringEvaluationOptions effective_options;
  bool batched = false;
  std::int64_t observation_count = 0;
  std::int64_t semantic_dimension_count = 0;
  std::vector<ClusteringEvaluationUnitResult> units;
};

// predicted:      [N] or [U,N], integral arbitrary cluster identifiers
// representable as int64. semantic_truth: [N,D] or [U,N,D], finite
// integral/floating semantic vectors. Exact groups are defined by full-vector
// equality. Mixed batched/unbatched ranks and empty axes are rejected. Results
// are evaluated independently per unit.
[[nodiscard]] ClusteringEvaluationResult
evaluate_clustering(const torch::Tensor &predicted,
                    const torch::Tensor &semantic_truth,
                    const ClusteringEvaluationOptions &options = {});

} // namespace leakflow::ml
