#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plot/table_view.hpp"
#include "leakflow/plugins/ml/clustering_evaluation_payload.hpp"
#include "leakflow/plugins/ml_plot/clustering_metrics_table_plot.hpp"

#include "leakflow/core/buffer.hpp"
#include "leakflow/ml/clustering_evaluation.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

template <typename Function> bool throws_invalid_argument(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

[[nodiscard]] leakflow::ml::MetricValue
defined_metric(leakflow::ml::ClusteringMetricId id, double value,
               std::uint64_t support) {
  return {
      .metric = id,
      .value = value,
      .support_count = support,
      .undefined_reason = leakflow::ml::MetricUndefinedReason::None,
  };
}

[[nodiscard]] leakflow::ml::MetricValue
undefined_metric(leakflow::ml::ClusteringMetricId id,
                 leakflow::ml::MetricUndefinedReason reason,
                 std::uint64_t support = 0) {
  return {
      .metric = id,
      .value = std::nullopt,
      .support_count = support,
      .undefined_reason = reason,
  };
}

[[nodiscard]] leakflow::ml::ClusteringEvaluationResult evaluation_result() {
  namespace ml = leakflow::ml;

  ml::ClusteringEvaluationResult result;
  result.effective_options.detail = ml::ClusteringEvaluationDetail::Full;
  result.effective_options.semantic = ml::SemanticEvaluationMode::Power;
  result.effective_options.dimension_names = {"hm", "hy",
                                              std::string(600, 'z')};
  result.effective_options.semantic_ranges = {2.0, 2.0};
  result.effective_options.semantic_weights = {3.0, 1.0};
  result.effective_options.power = 2;
  result.effective_options.alignment = ml::AlignmentEvaluationMode::Both;
  result.effective_options.combined_quality = true;
  result.observation_count = 6;
  result.semantic_dimension_count = 2;

  ml::ClusteringEvaluationUnitResult unit;
  unit.observation_count = 6;
  unit.truth_group_count = 3;
  unit.predicted_cluster_count = 3;
  unit.exact.adjusted_rand_index =
      defined_metric(ml::ClusteringMetricId::AdjustedRandIndex, 0.91, 15);
  unit.exact.adjusted_mutual_information = defined_metric(
      ml::ClusteringMetricId::AdjustedMutualInformation, 0.82, 6);
  unit.exact.homogeneity =
      defined_metric(ml::ClusteringMetricId::Homogeneity, 0.83, 6);
  unit.exact.completeness =
      defined_metric(ml::ClusteringMetricId::Completeness, 0.84, 6);
  unit.exact.v_measure =
      defined_metric(ml::ClusteringMetricId::VMeasure, 0.85, 6);
  unit.exact.purity = defined_metric(ml::ClusteringMetricId::Purity, 0.86, 6);
  unit.exact.pair_precision =
      defined_metric(ml::ClusteringMetricId::PairPrecision, 0.87, 6);
  unit.exact.pair_recall =
      defined_metric(ml::ClusteringMetricId::PairRecall, 0.88, 6);
  unit.exact.pair_f1 = defined_metric(ml::ClusteringMetricId::PairF1, 0.89, 12);
  unit.exact.normalized_mutual_information = defined_metric(
      ml::ClusteringMetricId::NormalizedMutualInformation, 0.90, 6);

  unit.semantic.micro_impurity =
      defined_metric(ml::ClusteringMetricId::SemanticImpurityMicro, 0.20, 12);
  unit.semantic.macro_impurity =
      defined_metric(ml::ClusteringMetricId::SemanticImpurityMacro, 0.22, 3);
  unit.semantic.merge_error_rate =
      defined_metric(ml::ClusteringMetricId::MergeErrorRate, 0.23, 6);
  unit.semantic.conditional_merge_error_severity =
      undefined_metric(ml::ClusteringMetricId::ConditionalMergeErrorSeverity,
                       ml::MetricUndefinedReason::NoMergeErrorPairs);
  unit.semantic.dimensions = {
      {
          .dimension_index = 0,
          .micro_impurity = defined_metric(
              ml::ClusteringMetricId::SemanticImpurityDimensionMicro, 0.24, 12),
          .macro_impurity = defined_metric(
              ml::ClusteringMetricId::SemanticImpurityDimensionMacro, 0.25, 3),
      },
      {
          .dimension_index = 1,
          .micro_impurity = defined_metric(
              ml::ClusteringMetricId::SemanticImpurityDimensionMicro, 0.26, 12),
          .macro_impurity = defined_metric(
              ml::ClusteringMetricId::SemanticImpurityDimensionMacro, 0.27, 3),
      },
  };
  unit.semantic.cluster_details =
      std::vector<ml::PredictedClusterSemanticDetail>{
          {
              .predicted_cluster_index = 0,
              .observation_count = 2,
              .impurity = defined_metric(
                  ml::ClusteringMetricId::SemanticImpurityPerCluster, 0.28, 1),
          },
      };

  unit.fragmentation.micro =
      defined_metric(ml::ClusteringMetricId::FragmentationMicro, 0.30, 12);
  unit.fragmentation.macro =
      defined_metric(ml::ClusteringMetricId::FragmentationMacro, 0.31, 3);
  unit.fragmentation.group_details =
      std::vector<ml::TruthGroupFragmentationDetail>{
          {
              .truth_group_index = 0,
              .observation_count = 2,
              .fragmentation = defined_metric(
                  ml::ClusteringMetricId::FragmentationPerGroup, 0.32, 1),
          },
      };

  ml::CombinedClusteringQuality combined;
  combined.quality =
      defined_metric(ml::ClusteringMetricId::CombinedQuality, 0.70, 6);
  combined.semantic_micro_impurity =
      defined_metric(ml::ClusteringMetricId::SemanticImpurityMicro, 0.21, 15);
  combined.fragmentation_micro =
      defined_metric(ml::ClusteringMetricId::FragmentationMicro, 0.33, 14);
  unit.combined_quality = std::move(combined);

  ml::ExactAlignmentTruthGroupDetail exact_group;
  exact_group.truth_group_index = 0;
  exact_group.predicted_cluster_index = 0;
  exact_group.truth_observation_count = 2;
  exact_group.predicted_observation_count = 2;
  exact_group.overlap_observation_count = 2;
  exact_group.precision = defined_metric(
      ml::ClusteringMetricId::ExactAlignmentPrecisionPerGroup, 0.93, 2);
  exact_group.recall = defined_metric(
      ml::ClusteringMetricId::ExactAlignmentRecallPerGroup, 0.94, 2);
  exact_group.f1 =
      defined_metric(ml::ClusteringMetricId::ExactAlignmentF1PerGroup, 0.95, 4);
  exact_group.jaccard = defined_metric(
      ml::ClusteringMetricId::ExactAlignmentJaccardPerGroup, 0.96, 2);
  ml::ExactOverlapAlignment exact_alignment;
  exact_alignment.matched_accuracy = defined_metric(
      ml::ClusteringMetricId::ExactAlignmentMatchedAccuracy, 0.92, 6);
  exact_alignment.truth_group_details =
      std::vector<ml::ExactAlignmentTruthGroupDetail>{std::move(exact_group)};
  unit.exact_alignment = std::move(exact_alignment);

  ml::SemanticCostAlignment semantic_alignment;
  semantic_alignment.normalized_cost =
      defined_metric(ml::ClusteringMetricId::SemanticAlignmentCost, 0.10, 6);
  semantic_alignment.dimensions = {
      {
          .dimension_index = 0,
          .normalized_error = defined_metric(
              ml::ClusteringMetricId::SemanticAlignmentDimensionError, 0.11, 6),
      },
      {
          .dimension_index = 1,
          .normalized_error = defined_metric(
              ml::ClusteringMetricId::SemanticAlignmentDimensionError, 0.12, 6),
      },
  };
  unit.semantic_alignment = std::move(semantic_alignment);

  result.units.push_back(std::move(unit));
  return result;
}

leakflow::Buffer evaluation_buffer(bool add_append_parameter = false) {
  leakflow::plugins::ml::ClusteringEvaluationPayload::Parameters parameters{
      {"evaluation.power", "999"},
      {"labels.cluster.covariance_type", "diagonal"},
      {"labels.cluster.n_components", "3"},
      {"metric", "payload-header-collision"},
  };
  auto payload =
      std::make_shared<leakflow::plugins::ml::ClusteringEvaluationPayload>(
          evaluation_result(), std::move(parameters));
  leakflow::Buffer buffer{
      leakflow::Caps(leakflow::plugins::ml::clustering_evaluation_caps_type)};
  buffer.set_units(leakflow::Units::of({5}));
  buffer.set_metadata("payload.parameter.dataset", "attack-key-05");
  buffer.set_metadata("payload.parameter.value", "metadata-header-collision");
  if (add_append_parameter) {
    buffer.set_metadata("payload.parameter.seed", "7");
  }
  buffer.set_payload(std::move(payload));
  return buffer;
}

std::size_t column_index(const leakflow::plot::TableSnapshot &snapshot,
                         const std::string &name) {
  const auto found =
      std::find(snapshot.columns.begin(), snapshot.columns.end(), name);
  if (found == snapshot.columns.end()) {
    throw std::runtime_error("missing table column " + name);
  }
  return static_cast<std::size_t>(
      std::distance(snapshot.columns.begin(), found));
}

bool has_metric(const leakflow::plot::TableSnapshot &snapshot,
                const std::string &name) {
  const auto metric = column_index(snapshot, "metric");
  const auto &rows = snapshot.frames.back().rows;
  return std::any_of(rows.begin(), rows.end(), [&](const auto &row) {
    return metric < row.size() && row[metric].text == name;
  });
}

bool column_equals(const leakflow::plot::TableSnapshot &snapshot,
                   const std::string &name, const std::string &expected) {
  const auto column = column_index(snapshot, name);
  const auto &rows = snapshot.frames.back().rows;
  return !rows.empty() &&
         std::all_of(rows.begin(), rows.end(), [&](const auto &row) {
           return column < row.size() && row[column].text == expected;
         });
}

const std::vector<leakflow::plot::TableCell> *
metric_row(const leakflow::plot::TableSnapshot &snapshot,
           const std::string &scope_name, const std::string &item_name,
           const std::string &metric_name) {
  const auto scope = column_index(snapshot, "scope");
  const auto item = column_index(snapshot, "item");
  const auto metric = column_index(snapshot, "metric");
  const auto &rows = snapshot.frames.back().rows;
  const auto found =
      std::find_if(rows.begin(), rows.end(), [&](const auto &row) {
        return row[scope].text == scope_name && row[item].text == item_name &&
               row[metric].text == metric_name;
      });
  return found == rows.end() ? nullptr : &*found;
}

} // namespace

int main() {
  namespace ml_plot = leakflow::plugins::ml_plot;
  namespace plot = leakflow::plot;

  const auto descriptor = ml_plot::ClusteringMetricsTablePlot::descriptor();
  if (!expect(descriptor.input_pads.size() == 1 &&
                  descriptor.input_pads.front().caps().type() ==
                      leakflow::plugins::ml::clustering_evaluation_caps_type &&
                  descriptor.output_pads.empty(),
              "table descriptor pads/caps are wrong")) {
    return 1;
  }
  const auto property_named = [&](const std::string &name) {
    return std::find_if(
        descriptor.property_specs.begin(), descriptor.property_specs.end(),
        [&](const auto &property) { return property.name == name; });
  };
  const auto group = property_named("group");
  const auto title = property_named("title");
  const auto update_mode = property_named("update_mode");
  if (!expect(group != descriptor.property_specs.end() &&
                  title != descriptor.property_specs.end() &&
                  update_mode != descriptor.property_specs.end(),
              "table descriptor properties are missing") ||
      !expect(group->effect.kind == leakflow::PropertyEffectKind::UiControl &&
                  title->effect.kind ==
                      leakflow::PropertyEffectKind::UiControl &&
                  group->effect.scope ==
                      leakflow::PropertyInvalidationScope::ElementUi &&
                  title->effect.scope ==
                      leakflow::PropertyInvalidationScope::ElementUi,
              "group/title controls must be UI-only") ||
      !expect(update_mode->effect.kind ==
                      leakflow::PropertyEffectKind::SinkDisplay &&
                  update_mode->effect.scope ==
                      leakflow::PropertyInvalidationScope::ElementUi,
              "update_mode must retranslate only the cached sink payload")) {
    return 1;
  }

  auto runtime = std::make_shared<plot::PlotRuntime>();
  auto view = std::make_shared<plot::TableView>();
  runtime->add_view(view);
  ml_plot::ClusteringMetricsTablePlot table(view, "metrics");
  if (!expect(!table.process(evaluation_buffer()).has_value(),
              "table sink must not emit a buffer")) {
    return 1;
  }
  auto snapshots = view->snapshots_copy();
  if (!expect(snapshots.size() == 1 && runtime->has_sessions(),
              "table did not register one visible snapshot")) {
    return 1;
  }
  const auto &snapshot = snapshots.front();
  if (!expect(snapshot.group == "clustering" && snapshot.frames.size() == 1,
              "table identity/default replace state is wrong")) {
    return 1;
  }
  if (!expect(
          column_index(snapshot,
                       "parameter.payload.labels.cluster.covariance_type") <
                  snapshot.columns.size() &&
              column_index(snapshot,
                           "parameter.payload.labels.cluster.n_components") <
                  snapshot.columns.size() &&
              column_index(snapshot, "parameter.metadata.dataset") <
                  snapshot.columns.size() &&
              column_index(snapshot, "parameter.payload.metric") <
                  snapshot.columns.size() &&
              column_index(snapshot, "parameter.metadata.value") <
                  snapshot.columns.size(),
          "payload/metadata parameters were not promoted to sortable "
          "collision-proof columns") ||
      !expect(std::set<std::string>(snapshot.columns.begin(),
                                    snapshot.columns.end())
                      .size() == snapshot.columns.size(),
              "parameter columns collided with fixed table headers")) {
    return 1;
  }
  if (!expect(
          column_equals(snapshot, "parameter.payload.evaluation.detail",
                        "full") &&
              column_equals(snapshot, "parameter.payload.evaluation.semantic",
                            "power") &&
              column_equals(snapshot,
                            "parameter.payload.evaluation.semantic_ranges",
                            "[2,2]") &&
              column_equals(snapshot,
                            "parameter.payload.evaluation.semantic_weights",
                            "[3,1]") &&
              column_equals(snapshot, "parameter.payload.evaluation.power",
                            "2") &&
              column_equals(snapshot, "parameter.payload.evaluation.alignment",
                            "both") &&
              column_equals(snapshot,
                            "parameter.payload.evaluation.combined_quality",
                            "true"),
          "effective result options were omitted or lost authority over "
          "captured payload parameters")) {
    return 1;
  }
  {
    const auto dimension_names =
        column_index(snapshot, "parameter.payload.evaluation.dimension_names");
    const auto &display =
        snapshot.frames.back().rows.front()[dimension_names].text;
    if (!expect(display.size() == 512 && display.starts_with("[hm,hy,"),
                "effective option presentation strings must be bounded to 512 "
                "characters")) {
      return 1;
    }
  }
  for (const auto &metric : leakflow::ml::clustering_metric_descriptors()) {
    if (!expect(has_metric(snapshot, std::string(metric.name)),
                "stored metric is missing from the table: " +
                    std::string(metric.name))) {
      return 1;
    }
  }
  if (!expect(
          snapshot.frames.back().rows.size() == 33,
          "the fixture's stored metric/detail rows were not all translated")) {
    return 1;
  }
  {
    const auto semantic_component =
        metric_row(snapshot, "combined_component", "semantic_micro_impurity",
                   "semantic_impurity_micro");
    const auto fragmentation_component =
        metric_row(snapshot, "combined_component", "fragmentation_micro",
                   "fragmentation_micro");
    const auto value = column_index(snapshot, "value");
    const auto support = column_index(snapshot, "support");
    if (!expect(semantic_component != nullptr &&
                    (*semantic_component)[value].text == "0.210000" &&
                    (*semantic_component)[support].text == "15" &&
                    fragmentation_component != nullptr &&
                    (*fragmentation_component)[value].text == "0.330000" &&
                    (*fragmentation_component)[support].text == "14",
                "combined-quality source MetricValues were not translated "
                "as explicit component rows")) {
      return 1;
    }
  }
  {
    const auto metric = column_index(snapshot, "metric");
    const auto value = column_index(snapshot, "value");
    const auto support = column_index(snapshot, "support");
    const auto direction = column_index(snapshot, "direction");
    const auto status = column_index(snapshot, "status");
    const auto &rows = snapshot.frames.back().rows;
    const auto found =
        std::find_if(rows.begin(), rows.end(), [&](const auto &row) {
          return row[metric].text == "conditional_merge_error_severity";
        });
    if (!expect(found != rows.end() && (*found)[value].text == "N/A" &&
                    (*found)[support].text == "0" &&
                    (*found)[direction].text == "lower_is_better" &&
                    (*found)[status].text == "no_merge_error_pairs" &&
                    !(*found)[value].hover.empty(),
                "undefined metric reason/support/direction display is wrong")) {
      return 1;
    }
  }

  // Replace is the default: a second result replaces the visible rows, while
  // append accumulates another complete run in the same sortable table.
  const auto first_row_count = snapshot.frames.back().rows.size();
  (void)table.process(evaluation_buffer());
  snapshots = view->snapshots_copy();
  if (!expect(snapshots.front().frames.back().rows.size() == first_row_count,
              "replace mode must replace current metric rows")) {
    return 1;
  }
  table.set_property("update_mode", std::string("append"));
  (void)table.process(evaluation_buffer(true));
  snapshots = view->snapshots_copy();
  const auto seed = column_index(snapshots.front(), "parameter.metadata.seed");
  const auto &appended_rows = snapshots.front().frames.back().rows;
  if (!expect(appended_rows.size() == first_row_count * 2,
              "append mode must accumulate a second evaluation") ||
      !expect(appended_rows.front()[seed].text.empty() &&
                  appended_rows[first_row_count][seed].text == "7",
              "append schema union did not backfill/add parameter columns")) {
    return 1;
  }

  // Group/title controls update presentation directly and do not replay the
  // payload or append duplicate rows.
  table.set_property("group", std::string("compare"));
  table.set_property("title", std::string("GMM comparison"));
  snapshots = view->snapshots_copy();
  if (!expect(snapshots.front().group == "compare" &&
                  snapshots.front().title == "GMM comparison" &&
                  snapshots.front().frames.back().rows.size() ==
                      first_row_count * 2,
              "presentation control changed data or failed to update")) {
    return 1;
  }

  // Per-producer clear leaves other tables intact; global runtime clear also
  // cascades through the view.
  auto second = plot::TableUpdate{};
  second.columns = {"value"};
  second.frame.rows = {{{.text = "other"}}};
  view->push("other", second);
  if (!expect(view->erase("metrics") && view->snapshots_copy().size() == 1,
              "per-table clear removed the wrong snapshots")) {
    return 1;
  }
  runtime->clear();
  if (!expect(view->empty() && !runtime->has_sessions(),
              "runtime clear did not clear the table view")) {
    return 1;
  }

  // Caps and payload validation are strict.
  leakflow::Buffer wrong{leakflow::Caps("leakflow/buffer")};
  if (!expect(throws_invalid_argument([&] { (void)table.process(wrong); }),
              "wrong input caps must be rejected")) {
    return 1;
  }
  if (!expect(
          throws_invalid_argument([&] { (void)table.process(std::nullopt); }),
          "missing input must be rejected")) {
    return 1;
  }

  return 0;
}
