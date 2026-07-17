#include "leakflow/plugins/ml/clustering_evaluation_payload.hpp"

#include "leakflow/core/summary_document.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace leakflow::plugins::ml {
namespace {

[[nodiscard]] std::string
detail_name(leakflow::ml::ClusteringEvaluationDetail value) {
  switch (value) {
  case leakflow::ml::ClusteringEvaluationDetail::Global:
    return "global";
  case leakflow::ml::ClusteringEvaluationDetail::Full:
    return "full";
  }
  return "unknown";
}

[[nodiscard]] std::string
semantic_name(leakflow::ml::SemanticEvaluationMode value) {
  switch (value) {
  case leakflow::ml::SemanticEvaluationMode::Off:
    return "off";
  case leakflow::ml::SemanticEvaluationMode::Power:
    return "power";
  }
  return "unknown";
}

[[nodiscard]] std::string
alignment_name(leakflow::ml::AlignmentEvaluationMode value) {
  switch (value) {
  case leakflow::ml::AlignmentEvaluationMode::None:
    return "none";
  case leakflow::ml::AlignmentEvaluationMode::Exact:
    return "exact";
  case leakflow::ml::AlignmentEvaluationMode::Semantic:
    return "semantic";
  case leakflow::ml::AlignmentEvaluationMode::Both:
    return "both";
  }
  return "unknown";
}

[[nodiscard]] std::string
format_metric(const leakflow::ml::MetricValue &metric) {
  if (!metric.value) {
    return "N/A (" +
           std::string(leakflow::ml::metric_undefined_reason_name(
               metric.undefined_reason)) +
           ")";
  }
  std::ostringstream output;
  output << std::fixed << std::setprecision(6) << *metric.value;
  return output.str();
}

void add_metric_child(SummaryField &field, std::string label,
                      const leakflow::ml::MetricValue &metric) {
  auto &child = field.add_child(std::move(label), format_metric(metric),
                                metric.value ? SummaryValueRole::Number
                                             : SummaryValueRole::Warning);
  child.add_child("support", summary_size(metric.support_count),
                  SummaryValueRole::Size);
}

} // namespace

ClusteringEvaluationPayload::ClusteringEvaluationPayload(
    leakflow::ml::ClusteringEvaluationResult result, Parameters parameters)
    : result_(std::move(result)), parameters_(std::move(parameters)) {
  if (result_.units.empty()) {
    throw std::invalid_argument(
        "ClusteringEvaluationPayload requires at least one unit result");
  }
  if (parameters_.size() > 64) {
    throw std::invalid_argument(
        "ClusteringEvaluationPayload accepts at most 64 bounded parameters");
  }
  for (const auto &[name, value] : parameters_) {
    if (name.empty() || name.size() > 128) {
      throw std::invalid_argument("ClusteringEvaluationPayload parameter names "
                                  "must contain 1 to 128 characters");
    }
    if (value.size() > 512) {
      throw std::invalid_argument("ClusteringEvaluationPayload parameter "
                                  "values may not exceed 512 characters");
    }
  }
}

std::string ClusteringEvaluationPayload::type_name() const {
  return clustering_evaluation_caps_type;
}

std::string ClusteringEvaluationPayload::layout() const {
  return "unit/evaluation";
}

void ClusteringEvaluationPayload::describe(SummarySection &section,
                                           std::int64_t summary_level) const {
  section.add_field("payload", type_name(), SummaryValueRole::TypeName);
  section.add_field("schema", summary_integer(result_.schema_version),
                    SummaryValueRole::Number);
  section.add_field(
      "units", summary_integer(static_cast<std::int64_t>(result_.units.size())),
      SummaryValueRole::Number);
  section.add_field("observations", summary_integer(result_.observation_count),
                    SummaryValueRole::Number);
  section.add_field("dimensions",
                    summary_integer(result_.semantic_dimension_count),
                    SummaryValueRole::Number);
  section.add_field("detail", detail_name(result_.effective_options.detail),
                    SummaryValueRole::Text);
  section.add_field("semantic",
                    semantic_name(result_.effective_options.semantic),
                    SummaryValueRole::Text);
  section.add_field("alignment",
                    alignment_name(result_.effective_options.alignment),
                    SummaryValueRole::Text);
  section.add_field(
      "parameters",
      summary_integer(static_cast<std::int64_t>(parameters_.size())),
      SummaryValueRole::Number);

  if (summary_level >= 1) {
    constexpr std::size_t max_summary_units = 8;
    const auto shown =
        std::min<std::size_t>(result_.units.size(), max_summary_units);
    for (std::size_t index = 0; index < shown; ++index) {
      const auto &unit = result_.units[index];
      auto &field = section.add_field(
          "unit[" + std::to_string(index) + "]",
          "groups=" + std::to_string(unit.truth_group_count) +
              " clusters=" + std::to_string(unit.predicted_cluster_count),
          SummaryValueRole::Number);
      add_metric_child(field, "adjusted_rand_index",
                       unit.exact.adjusted_rand_index);
      add_metric_child(field, "adjusted_mutual_information",
                       unit.exact.adjusted_mutual_information);
      add_metric_child(field, "v_measure", unit.exact.v_measure);
      add_metric_child(field, "purity", unit.exact.purity);
      add_metric_child(field, "pair_precision", unit.exact.pair_precision);
      add_metric_child(field, "pair_recall", unit.exact.pair_recall);
      add_metric_child(field, "pair_f1", unit.exact.pair_f1);
      add_metric_child(field, "semantic_impurity_micro",
                       unit.semantic.micro_impurity);
      add_metric_child(field, "semantic_partition_separation",
                       unit.semantic.partition_separation);
      add_metric_child(field, "fragmentation_micro", unit.fragmentation.micro);
      if (unit.semantic_partition_quality) {
        add_metric_child(field, "semantic_partition_quality",
                         unit.semantic_partition_quality->quality);
      }
      if (unit.combined_quality) {
        // Keep the established summary key for compatibility. The evaluator
        // property and plot label document that this score is legacy.
        add_metric_child(field, "combined_quality",
                         unit.combined_quality->quality);
      }
      if (unit.exact_alignment) {
        add_metric_child(field, "exact_alignment_matched_accuracy",
                         unit.exact_alignment->matched_accuracy);
      }
      if (unit.semantic_alignment) {
        add_metric_child(field, "semantic_alignment_cost",
                         unit.semantic_alignment->normalized_cost);
      }
    }
    if (shown < result_.units.size()) {
      section.add_field("units_not_shown",
                        summary_integer(static_cast<std::int64_t>(
                            result_.units.size() - shown)),
                        SummaryValueRole::Number);
    }
  }

  if (summary_level >= 2) {
    auto &parameters = section.add_field("parameter_values", "bounded",
                                         SummaryValueRole::Text);
    std::size_t shown = 0;
    for (const auto &[name, value] : parameters_) {
      if (shown++ == 16) {
        parameters.add_child(
            "remaining",
            summary_integer(static_cast<std::int64_t>(parameters_.size() - 16)),
            SummaryValueRole::Number);
        break;
      }
      parameters.add_child(name, value, SummaryValueRole::Text);
    }
  }
}

const leakflow::ml::ClusteringEvaluationResult &
ClusteringEvaluationPayload::result() const noexcept {
  return result_;
}

const ClusteringEvaluationPayload::Parameters &
ClusteringEvaluationPayload::parameters() const noexcept {
  return parameters_;
}

} // namespace leakflow::plugins::ml
