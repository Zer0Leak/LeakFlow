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
  AdjustedRandIndex = 0,
  AdjustedMutualInformation = 1,
  Homogeneity = 2,
  Completeness = 3,
  VMeasure = 4,
  Purity = 5,
  PairPrecision = 6,
  PairRecall = 7,
  PairF1 = 8,
  NormalizedMutualInformation = 9,
  SemanticImpurityMicro = 10,
  SemanticImpurityMacro = 11,
  MergeErrorRate = 12,
  ConditionalMergeErrorSeverity = 13,
  SemanticImpurityDimensionMicro = 14,
  SemanticImpurityDimensionMacro = 15,
  SemanticImpurityPerCluster = 16,
  FragmentationMicro = 17,
  FragmentationMacro = 18,
  FragmentationPerGroup = 19,
  ExactAlignmentMatchedAccuracy = 20,
  ExactAlignmentPrecisionPerGroup = 21,
  ExactAlignmentRecallPerGroup = 22,
  ExactAlignmentF1PerGroup = 23,
  ExactAlignmentJaccardPerGroup = 24,
  SemanticAlignmentCost = 25,
  SemanticAlignmentDimensionError = 26,
  CombinedQuality = 27,
  SemanticPartitionSeparation = 28,
  SemanticPartitionQuality = 29,
};

enum class MetricFamily : std::uint8_t {
  Exact,
  Semantic,
  Fragmentation,
  Alignment,
  Combined,
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

// Descriptors are stable, process-lifetime data. IDs and descriptor order are
// append-only so a later persistence codec can use the numeric IDs safely.
[[nodiscard]] const ClusteringMetricDescriptor &
clustering_metric_descriptor(ClusteringMetricId id);
[[nodiscard]] std::span<const ClusteringMetricDescriptor>
clustering_metric_descriptors();
[[nodiscard]] std::span<const ClusteringMetricDescriptor>
exact_clustering_metric_descriptors();
[[nodiscard]] std::string_view metric_family_name(MetricFamily family);
[[nodiscard]] std::string_view metric_direction_name(MetricDirection direction);
[[nodiscard]] std::string_view metric_averaging_name(MetricAveraging averaging);

enum class MetricUndefinedReason : std::uint8_t {
  None = 0,
  NoPredictedWithinClusterPairs = 1,
  NoTrueWithinGroupPairs = 2,
  DependentMetricUndefined = 3,
  SemanticDisabled = 4,
  NoEligiblePredictedClusters = 5,
  NoMergeErrorPairs = 6,
  NoEligibleTruthGroups = 7,
  NoMatchedPredictedCluster = 8,
  NoSemanticVariation = 9,
};

[[nodiscard]] std::string_view
metric_undefined_reason_name(MetricUndefinedReason reason);

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

struct SemanticDimensionMetrics {
  std::int64_t dimension_index = 0;
  MetricValue micro_impurity;
  MetricValue macro_impurity;
};

struct PredictedClusterSemanticDetail {
  // Dense index into ClusteringPartitionDetail::predicted_ids.
  std::int64_t predicted_cluster_index = 0;
  std::uint64_t observation_count = 0;
  MetricValue impurity;
};

struct SemanticClusteringMetrics {
  MetricValue micro_impurity;
  MetricValue macro_impurity;
  MetricValue merge_error_rate;
  MetricValue conditional_merge_error_severity;
  // Fraction of the dataset's total pairwise semantic cost that lies between
  // predicted clusters: 1 - D_within / D_all. Unlike micro_impurity, this is
  // zero for a one-cluster collapse and one when all nonzero semantic cost is
  // separated by predicted-cluster boundaries.
  MetricValue partition_separation;
  std::vector<SemanticDimensionMetrics> dimensions;

  // nullopt for Global detail; engaged for Full detail. Singleton clusters are
  // retained with an explicit undefined impurity record.
  std::optional<std::vector<PredictedClusterSemanticDetail>> cluster_details;
};

struct TruthGroupFragmentationDetail {
  // Dense index into ClusteringPartitionDetail::truth_vectors.
  std::int64_t truth_group_index = 0;
  std::uint64_t observation_count = 0;
  MetricValue fragmentation;
};

struct FragmentationClusteringMetrics {
  MetricValue micro;
  MetricValue macro;

  // nullopt for Global detail; engaged for Full detail. Singleton truth groups
  // are retained with an explicit undefined fragmentation record.
  std::optional<std::vector<TruthGroupFragmentationDetail>> group_details;
};

struct CombinedClusteringQuality {
  MetricValue quality;
  // Source records are copied verbatim so their distinct pair supports and
  // undefined reasons remain explicit beside the derived score.
  MetricValue semantic_micro_impurity;
  MetricValue fragmentation_micro;
};

struct SemanticPartitionClusteringQuality {
  MetricValue quality;
  // Source records are copied verbatim so their distinct supports and
  // undefined reasons remain explicit beside the derived score.
  MetricValue semantic_partition_separation;
  MetricValue pair_recall;
};

inline constexpr std::int64_t kUnmatchedAlignmentIndex = -1;

struct AlignmentUnmatchedSupport {
  // Dense truth-group or predicted-cluster index, according to the containing
  // vector.
  std::int64_t index = 0;
  std::uint64_t observation_count = 0;
};

struct ClusteringAlignmentMapping {
  // Canonical dense indices. Unmatched entries use
  // kUnmatchedAlignmentIndex. See ClusteringAlignmentIdentities for the
  // corresponding original predicted IDs and semantic truth vectors. Among
  // equal primary optima, both alignment methods choose the lexicographically
  // smallest predicted_to_truth_group vector in canonical predicted-ID order,
  // with real truth indices ordered before unmatched entries.
  std::vector<std::int64_t> predicted_to_truth_group;   // [K]
  std::vector<std::int64_t> truth_to_predicted_cluster; // [G]
  std::vector<AlignmentUnmatchedSupport> unmatched_truth_groups;
  std::vector<AlignmentUnmatchedSupport> unmatched_predicted_clusters;

  // Predicted-side and truth-side assignment supports are separate marginals;
  // neither pair should be interpreted as an observation partition shared by
  // both sides.
  std::uint64_t assigned_predicted_observation_count = 0;
  std::uint64_t unmatched_predicted_observation_count = 0;
  std::uint64_t assigned_truth_observation_count = 0;
  std::uint64_t unmatched_truth_observation_count = 0;
};

struct ExactAlignmentTruthGroupDetail {
  std::int64_t truth_group_index = 0;
  std::int64_t predicted_cluster_index = kUnmatchedAlignmentIndex;
  std::uint64_t truth_observation_count = 0;
  std::uint64_t predicted_observation_count = 0;
  std::uint64_t overlap_observation_count = 0;
  MetricValue precision;
  MetricValue recall;
  MetricValue f1;
  MetricValue jaccard;
};

struct ExactOverlapAlignment {
  ClusteringAlignmentMapping mapping;
  MetricValue matched_accuracy;
  std::uint64_t matched_overlap_observation_count = 0;
  std::uint64_t nonmatched_observation_count = 0;

  // New/aligned column position -> canonical predicted-cluster index. Matched
  // clusters follow truth-group order; unmatched clusters follow in ascending
  // canonical order.
  std::vector<std::int64_t> aligned_column_to_predicted_cluster;

  // nullopt for Global detail; one record per truth group for Full detail.
  std::optional<std::vector<ExactAlignmentTruthGroupDetail>>
      truth_group_details;
};

struct SemanticAlignmentDimensionError {
  std::int64_t dimension_index = 0;
  MetricValue normalized_error;
};

struct SemanticAlignmentErrorMass {
  // One deterministic record per nonzero contingency cell. The record carries
  // observation mass without expanding observations individually.
  std::int64_t source_truth_group_index = 0;
  std::int64_t predicted_cluster_index = 0;
  std::int64_t assigned_truth_group_index = kUnmatchedAlignmentIndex;
  std::uint64_t observation_count = 0;
  double normalized_cost = 0.0;
  std::vector<double> dimension_costs;
};

struct SemanticCostAlignment {
  ClusteringAlignmentMapping mapping;
  MetricValue normalized_cost;
  std::vector<SemanticAlignmentDimensionError> dimensions;

  // Sum of contingency overlap under this semantic mapping. This is reported
  // separately from the predicted/truth marginal supports in mapping.
  std::uint64_t exact_overlap_observation_count = 0;
  std::uint64_t nonoverlap_observation_count = 0;

  // A predicted cluster assigned to a dummy truth group pays this normalized
  // cost for every observation, in aggregate and in every dimension.
  double unmatched_predicted_penalty = 1.0;

  // nullopt for Global detail; deterministic contingency-mass records for Full
  // detail.
  std::optional<std::vector<SemanticAlignmentErrorMass>> error_masses;
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

struct ClusteringAlignmentIdentities {
  // Materialized whenever an alignment is requested, including Global detail,
  // so dense mapping indices remain self-describing.
  torch::Tensor predicted_ids; // [K], CPU int64
  torch::Tensor truth_vectors; // [G,D], CPU, original numeric dtype
};

struct ClusteringEvaluationUnitResult {
  std::int64_t observation_count = 0;
  std::int64_t truth_group_count = 0;
  std::int64_t predicted_cluster_count = 0;
  PairCounts pair_counts;
  ExactClusteringMetrics exact;
  SemanticClusteringMetrics semantic;
  FragmentationClusteringMetrics fragmentation;
  // Legacy score; absent unless
  // ClusteringEvaluationOptions::combined_quality is enabled.
  std::optional<CombinedClusteringQuality> combined_quality;
  // Preferred cross-K semantic score; absent unless
  // ClusteringEvaluationOptions::semantic_partition_quality is enabled.
  std::optional<SemanticPartitionClusteringQuality> semantic_partition_quality;
  std::optional<ClusteringAlignmentIdentities> alignment_identities;
  std::optional<ExactOverlapAlignment> exact_alignment;
  std::optional<SemanticCostAlignment> semantic_alignment;
  std::optional<ClusteringPartitionDetail> partition_detail;
};

enum class ClusteringEvaluationDetail : std::uint8_t {
  Global,
  Full,
};

enum class SemanticEvaluationMode : std::uint8_t {
  Off,
  Power,
};

enum class AlignmentEvaluationMode : std::uint8_t {
  None,
  Exact,
  Semantic,
  Both,
};

struct ClusteringEvaluationOptions {
  ClusteringEvaluationDetail detail = ClusteringEvaluationDetail::Global;
  std::vector<std::string> dimension_names;
  SemanticEvaluationMode semantic = SemanticEvaluationMode::Off;
  // Power mode requires one finite positive range per semantic dimension.
  // Empty weights mean equal weights; non-empty weights must be finite,
  // non-negative, and contain at least one positive entry. Supplied semantic
  // configuration is validated even when semantic evaluation is off.
  std::vector<double> semantic_ranges;
  std::vector<double> semantic_weights;
  std::int64_t power = 2;
  AlignmentEvaluationMode alignment = AlignmentEvaluationMode::None;
  // Deprecated legacy composite retained for result compatibility. It is not
  // suitable for comparing different predicted-cluster counts.
  bool combined_quality = false;
  bool semantic_partition_quality = false;
};

struct ClusteringEvaluationResult {
  std::uint32_t schema_version = 5;
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
