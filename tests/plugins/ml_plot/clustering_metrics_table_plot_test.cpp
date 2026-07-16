#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plot/table_view.hpp"
#include "leakflow/plugins/ml/clustering_evaluation_payload.hpp"
#include "leakflow/plugins/ml_plot/clustering_metrics_table_plot.hpp"

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/ml/clustering_evaluation.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace ml = leakflow::ml;
namespace ml_plugin = leakflow::plugins::ml;
namespace ml_plot = leakflow::plugins::ml_plot;
namespace plot = leakflow::plot;

class LiveEvaluationSource final : public leakflow::Element {
public:
  LiveEvaluationSource() : Element("live-evaluation") {
    configure_from_descriptor({
        .type_name = "LiveEvaluationSource",
        .klass = "Source/Test",
        .output_pads =
            {
                leakflow::Pad(
                    "src", leakflow::PadDirection::Output,
                    leakflow::Caps(ml_plugin::clustering_evaluation_caps_type)),
            },
        .live_source = true,
    });
  }

  [[nodiscard]] std::optional<leakflow::Buffer>
  process(std::optional<leakflow::Buffer>) override {
    return std::nullopt;
  }
};

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

[[nodiscard]] ml::MetricValue
defined_metric(ml::ClusteringMetricId id, double value, std::uint64_t support) {
  return {
      .metric = id,
      .value = value,
      .support_count = support,
      .undefined_reason = ml::MetricUndefinedReason::None,
  };
}

[[nodiscard]] ml::MetricValue undefined_metric(ml::ClusteringMetricId id,
                                               ml::MetricUndefinedReason reason,
                                               std::uint64_t support = 0) {
  return {
      .metric = id,
      .value = std::nullopt,
      .support_count = support,
      .undefined_reason = reason,
  };
}

[[nodiscard]] ml::ClusteringEvaluationResult evaluation_result() {
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

[[nodiscard]] ml::ClusteringEvaluationResult result_without_optional_modes() {
  auto result = evaluation_result();
  result.effective_options.semantic = ml::SemanticEvaluationMode::Off;
  result.effective_options.alignment = ml::AlignmentEvaluationMode::None;
  result.effective_options.combined_quality = false;
  auto &unit = result.units.front();
  unit.combined_quality.reset();
  unit.exact_alignment.reset();
  unit.semantic_alignment.reset();

  const auto disabled = ml::MetricUndefinedReason::SemanticDisabled;
  unit.semantic.micro_impurity =
      undefined_metric(ml::ClusteringMetricId::SemanticImpurityMicro, disabled);
  unit.semantic.macro_impurity =
      undefined_metric(ml::ClusteringMetricId::SemanticImpurityMacro, disabled);
  unit.semantic.merge_error_rate =
      undefined_metric(ml::ClusteringMetricId::MergeErrorRate, disabled);
  unit.semantic.conditional_merge_error_severity = undefined_metric(
      ml::ClusteringMetricId::ConditionalMergeErrorSeverity, disabled);
  for (auto &dimension : unit.semantic.dimensions) {
    dimension.micro_impurity = undefined_metric(
        ml::ClusteringMetricId::SemanticImpurityDimensionMicro, disabled);
    dimension.macro_impurity = undefined_metric(
        ml::ClusteringMetricId::SemanticImpurityDimensionMacro, disabled);
  }
  if (unit.semantic.cluster_details) {
    for (auto &cluster : *unit.semantic.cluster_details) {
      cluster.impurity = undefined_metric(
          ml::ClusteringMetricId::SemanticImpurityPerCluster, disabled);
    }
  }
  return result;
}

[[nodiscard]] leakflow::Buffer
evaluation_buffer(ml::ClusteringEvaluationResult result = evaluation_result(),
                  bool add_seed = false) {
  ml_plugin::ClusteringEvaluationPayload::Parameters parameters{
      {"evaluation.power", "999"},
      {"evaluation.note", "future-option"},
      {"labels.cluster.covariance_type", "diagonal"},
      {"labels.cluster.n_components", "3"},
      {"metric", "payload-header-collision"},
  };
  auto payload = std::make_shared<ml_plugin::ClusteringEvaluationPayload>(
      std::move(result), std::move(parameters));
  leakflow::Buffer buffer{
      leakflow::Caps(ml_plugin::clustering_evaluation_caps_type)};
  buffer.set_units(leakflow::Units::of({5}));
  buffer.set_metadata("payload.parameter.dataset", "attack-key-05");
  buffer.set_metadata("payload.parameter.value", "metadata-header-collision");
  buffer.set_metadata("payload.parameter.covariance_type", "experiment-full");
  if (add_seed) {
    buffer.set_metadata("payload.parameter.seed", "7");
  }
  buffer.set_payload(std::move(payload));
  return buffer;
}

[[nodiscard]] std::size_t column_index(const plot::TableSnapshot &snapshot,
                                       std::string_view name) {
  const auto found = std::ranges::find(snapshot.columns, name);
  if (found == snapshot.columns.end()) {
    throw std::runtime_error("missing table column " + std::string(name));
  }
  return static_cast<std::size_t>(
      std::distance(snapshot.columns.begin(), found));
}

[[nodiscard]] const plot::TableSnapshot &
snapshot_for(const std::vector<plot::TableSnapshot> &snapshots,
             std::string_view label) {
  const auto found =
      std::ranges::find_if(snapshots, [label](const auto &value) {
        return value.tab_label == label;
      });
  if (found == snapshots.end()) {
    throw std::runtime_error("missing table tab " + std::string(label));
  }
  return *found;
}

[[nodiscard]] bool
has_snapshot(const std::vector<plot::TableSnapshot> &snapshots,
             std::string_view label) {
  return std::ranges::any_of(snapshots, [label](const auto &value) {
    return value.tab_label == label;
  });
}

[[nodiscard]] std::optional<std::string>
hover_value(const plot::TableCell &cell, std::string_view key) {
  const auto found = std::ranges::find_if(
      cell.hover, [key](const auto &entry) { return entry.first == key; });
  if (found == cell.hover.end()) {
    return std::nullopt;
  }
  return found->second;
}

[[nodiscard]] const std::vector<plot::TableCell> *
row_with(const plot::TableSnapshot &snapshot, std::string_view column,
         std::string_view value) {
  const auto index = column_index(snapshot, column);
  const auto &rows = snapshot.frames.back().rows;
  const auto found = std::ranges::find_if(rows, [&](const auto &row) {
    return index < row.size() && row[index].text == value;
  });
  return found == rows.end() ? nullptr : &*found;
}

[[nodiscard]] std::size_t
total_metric_rows(const std::vector<plot::TableSnapshot> &snapshots) {
  std::size_t count = 0;
  for (const auto label :
       {"Exact", "Semantic", "Fragmentation", "Combined", "Alignment"}) {
    count += snapshot_for(snapshots, label).frames.back().rows.size();
  }
  return count;
}

} // namespace

int main() {
  const auto descriptor = ml_plot::ClusteringMetricsTablePlot::descriptor();
  const auto property_named = [&](std::string_view name) {
    return std::ranges::find_if(
        descriptor.property_specs,
        [name](const auto &property) { return property.name == name; });
  };
  const auto group = property_named("group");
  const auto title = property_named("title");
  const auto update_mode = property_named("update_mode");
  const auto active_update_mode = property_named("active_update_mode");
  if (!expect(descriptor.input_pads.size() == 1 &&
                  descriptor.input_pads.front().caps().type() ==
                      ml_plugin::clustering_evaluation_caps_type &&
                  descriptor.output_pads.empty(),
              "table descriptor pads/caps are wrong") ||
      !expect(group != descriptor.property_specs.end() &&
                  title != descriptor.property_specs.end() &&
                  update_mode != descriptor.property_specs.end() &&
                  active_update_mode != descriptor.property_specs.end(),
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
                      leakflow::PropertyEffectKind::UiControl &&
                  update_mode->effect.scope ==
                      leakflow::PropertyInvalidationScope::ElementUi,
              "update_mode must be an element UI control") ||
      !expect(
          std::get<std::string>(update_mode->default_value) == "auto" &&
              std::get<leakflow::StringEnumConstraint>(update_mode->constraint)
                      .allowed_values ==
                  std::vector<std::string>({"auto", "accumulate", "replace"}),
          "update_mode default or allowed values are wrong") ||
      !expect(std::get<std::string>(active_update_mode->default_value) ==
                      "replace" &&
                  std::get<leakflow::StringEnumConstraint>(
                      active_update_mode->constraint)
                          .allowed_values ==
                      std::vector<std::string>({"accumulate", "replace"}) &&
                  !active_update_mode->writable &&
                  active_update_mode->effect.kind ==
                      leakflow::PropertyEffectKind::UiControl &&
                  active_update_mode->effect.scope ==
                      leakflow::PropertyInvalidationScope::None,
              "active_update_mode contract is wrong")) {
    return 1;
  }

  // Stopped-state changes resolve immediately and survive start(). The
  // read-only mirror cannot be changed through the public property API.
  {
    auto stopped_view = std::make_shared<plot::TableView>();
    ml_plot::ClusteringMetricsTablePlot stopped(stopped_view, "stopped");
    stopped.set_property("update_mode", std::string("accumulate"));
    if (!expect(stopped.property_as<std::string>("active_update_mode") ==
                        std::optional<std::string>("accumulate") &&
                    stopped_view->empty(),
                "stopped update_mode did not resolve without processing") ||
        !expect(throws_invalid_argument([&] {
                  stopped.set_property("active_update_mode",
                                       std::string("replace"));
                }),
                "active_update_mode accepted a public write")) {
      return 1;
    }
    stopped.start();
    if (!expect(stopped.property_as<std::string>("active_update_mode") ==
                    std::optional<std::string>("accumulate"),
                "start discarded a stopped-state update_mode")) {
      return 1;
    }
    (void)stopped.process(evaluation_buffer());
    (void)stopped.process(evaluation_buffer());
    const auto stopped_snapshots = stopped_view->snapshots_copy();
    if (!expect(
            snapshot_for(stopped_snapshots, "Overview")
                        .frames.back()
                        .rows.size() == 2 &&
                snapshot_for(stopped_snapshots, "Exact").frames.size() == 2,
            "stopped-state accumulate was not used by subsequent buffers")) {
      return 1;
    }
  }

  // Auto follows graph liveness, while explicit values continue to override
  // it. Linking is enough to propagate liveness; no GUI or pipeline run is
  // needed for this contract check.
  {
    leakflow::Pipeline pipeline;
    const auto live = pipeline.add(std::make_shared<LiveEvaluationSource>());
    auto live_table = std::make_shared<ml_plot::ClusteringMetricsTablePlot>(
        std::make_shared<plot::TableView>(), "live-table");
    const auto sink = pipeline.add(live_table);
    pipeline.link(live, "src", sink, "sink");
    if (!expect(
            pipeline.is_live_driven(sink) &&
                live_table->property_as<std::string>("active_update_mode") ==
                    std::optional<std::string>("accumulate"),
            "auto did not resolve to accumulate for live input")) {
      return 1;
    }
    live_table->set_property("update_mode", std::string("replace"));
    if (!expect(live_table->property_as<std::string>("active_update_mode") ==
                    std::optional<std::string>("replace"),
                "explicit replace did not override live liveness")) {
      return 1;
    }
    live_table->set_property("update_mode", std::string("auto"));
    if (!expect(live_table->property_as<std::string>("active_update_mode") ==
                    std::optional<std::string>("accumulate"),
                "auto did not restore live accumulation")) {
      return 1;
    }
  }

  // Multi-unit payloads retain the typed Buffer.units() identities and expose
  // one synchronized, view-local Unit selector on every unit-bearing tab.
  // Parameters are run-wide and deliberately remain unfiltered.
  {
    auto multi_result = evaluation_result();
    multi_result.units.push_back(multi_result.units.front());
    multi_result.units.back().combined_quality.reset();
    multi_result.units.back().exact_alignment.reset();
    multi_result.units.back().semantic_alignment.reset();
    auto multi_buffer = evaluation_buffer(std::move(multi_result));
    multi_buffer.set_units(leakflow::Units::of({2, 7}));
    auto multi_view = std::make_shared<plot::TableView>();
    ml_plot::ClusteringMetricsTablePlot multi(multi_view, "multi");
    (void)multi.process(std::move(multi_buffer));
    const auto multi_snapshots = multi_view->snapshots_copy();
    const auto &multi_overview = snapshot_for(multi_snapshots, "Overview");
    const auto unit_column = column_index(multi_overview, "Unit");
    if (!expect(
            multi_overview.frames.back().rows.size() == 2 &&
                multi_overview.frames.back().rows[0][unit_column].text == "2" &&
                multi_overview.frames.back().rows[1][unit_column].text == "7",
            "multi-unit rows lost dense-to-logical unit identity")) {
      return 1;
    }
    if (!expect(multi_view->select_row_value("multi.unit", std::int64_t{7}),
                "Unit selector rejected a declared logical unit")) {
      return 1;
    }
    for (const auto &snapshot : multi_snapshots) {
      if (snapshot.tab_label == "Parameters") {
        if (!expect(!snapshot.row_selector.has_value(),
                    "run-wide Parameters unexpectedly has a Unit selector")) {
          return 1;
        }
        continue;
      }
      if (!expect(snapshot.row_selector.has_value() &&
                      snapshot.row_selector->key == "multi.unit" &&
                      snapshot.row_selector->label == "Unit" &&
                      snapshot.row_selector->column == "Unit" &&
                      snapshot.row_selector->values.size() == 2 &&
                      snapshot.row_selector->values[0].value ==
                          plot::TableCell::SortValue(std::int64_t{2}) &&
                      snapshot.row_selector->values[1].value ==
                          plot::TableCell::SortValue(std::int64_t{7}),
                  "unit-bearing tabs do not share the Unit selector")) {
        return 1;
      }
      const auto family_unit_column = column_index(snapshot, "Unit");
      std::set<std::string> displayed_units;
      for (const auto &row : snapshot.frames.back().rows) {
        displayed_units.insert(row[family_unit_column].text);
      }
      const auto optional_subset =
          snapshot.tab_label == "Combined" || snapshot.tab_label == "Alignment";
      const auto expected_units = optional_subset
                                      ? std::set<std::string>({"2"})
                                      : std::set<std::string>({"2", "7"});
      std::size_t selected_rows = 0;
      for (std::size_t row = 0; row < snapshot.frames.back().rows.size();
           ++row) {
        selected_rows += plot::table_row_matches_selector(
                             snapshot.frames.back(), row, family_unit_column,
                             std::int64_t{7})
                             ? 1
                             : 0;
      }
      if (!expect(
              displayed_units == expected_units &&
                  (optional_subset ? selected_rows == 0 : selected_rows > 0),
              "a unit-bearing tab has wrong logical-unit filtering") ||
          !expect(
              multi_view->selected_row_value("multi.unit") ==
                  std::optional<plot::TableCell::SortValue>(std::int64_t{7}),
              "a sparse optional tab changed the shared Unit selection")) {
        return 1;
      }
    }
  }

  // The output Buffer's typed unit axis must map one-to-one to result units;
  // silently mixing typed and dense fallback identities would make the slider
  // merge or mislabel pages.
  {
    auto malformed_result = evaluation_result();
    malformed_result.units.push_back(malformed_result.units.front());
    auto malformed = evaluation_buffer(std::move(malformed_result));
    malformed.set_units(leakflow::Units::of({5}));
    ml_plot::ClusteringMetricsTablePlot malformed_table(
        std::make_shared<plot::TableView>(), "malformed");
    if (!expect(throws_invalid_argument([&] {
                  (void)malformed_table.process(std::move(malformed));
                }),
                "mismatched Buffer.units cardinality was accepted")) {
      return 1;
    }
  }

  auto runtime = std::make_shared<plot::PlotRuntime>();
  auto view = std::make_shared<plot::TableView>();
  runtime->add_view(view);
  ml_plot::ClusteringMetricsTablePlot table(view, "metrics");
  if (!expect(!table.process(evaluation_buffer()).has_value(),
              "table sink must not emit a buffer") ||
      !expect(table.property_as<std::string>("active_update_mode") ==
                  std::optional<std::string>("replace"),
              "auto did not resolve to replace for an offline table")) {
    return 1;
  }

  auto snapshots = view->snapshots_copy();
  const std::vector<std::string> expected_labels{
      "Overview", "Exact",     "Semantic",  "Fragmentation",
      "Combined", "Alignment", "Parameters"};
  if (!expect(snapshots.size() == expected_labels.size() &&
                  runtime->has_sessions(),
              "rich evaluation did not create all seven tabs")) {
    return 1;
  }
  for (std::size_t index = 0; index < snapshots.size(); ++index) {
    const auto &snapshot = snapshots[index];
    if (!expect(snapshot.tab_label == expected_labels[index] &&
                    snapshot.element_name ==
                        "metrics." + expected_labels[index] &&
                    snapshot.group == "clustering" &&
                    snapshot.title == "Clustering evaluation metrics" &&
                    snapshot.group_layout == plot::TableGroupLayout::Tabs &&
                    snapshot.tab_order == static_cast<int>(index) &&
                    snapshot.frames.size() == 1,
                "tab identity/layout/insertion order is wrong for " +
                    expected_labels[index])) {
      return 1;
    }
  }

  // Overview is intentionally compact: one row per run/unit, readable counts,
  // selected clustering/experiment parameters, and the six headline metrics.
  const auto &overview = snapshot_for(snapshots, "Overview");
  const auto &overview_row = overview.frames.back().rows.front();
  if (!expect(
          overview.frames.back().rows.size() == 1 &&
              overview_row[column_index(overview, "Run")].text == "1" &&
              overview_row[column_index(overview, "Unit")].text == "5" &&
              overview_row[column_index(overview, "Observations")].text ==
                  "6" &&
              overview_row[column_index(overview, "Truth groups")].text ==
                  "3" &&
              overview_row[column_index(overview, "Predicted clusters")].text ==
                  "3",
          "overview run/unit/count context is wrong") ||
      !expect(
          overview_row[column_index(overview, "Clustering: Covariance Type")]
                      .text == "diagonal" &&
              overview_row[column_index(overview, "Clustering: N Components")]
                      .text == "3" &&
              overview_row[column_index(overview, "Experiment: Dataset")]
                      .text == "attack-key-05" &&
              overview_row[column_index(overview,
                                        "Experiment: Covariance Type")]
                      .text == "experiment-full",
          "overview parameter selection or same-suffix separation is wrong") ||
      !expect(std::ranges::none_of(overview.columns,
                                   [](const auto &column) {
                                     return column.starts_with("Evaluator") ||
                                            column.starts_with("Evaluation");
                                   }),
              "verbose evaluator options leaked into Overview") ||
      !expect(
          overview_row[column_index(overview, "ARI \u2191")].text ==
                  "0.910000" &&
              overview_row[column_index(overview, "AMI \u2191")].text ==
                  "0.820000" &&
              overview_row[column_index(overview, "Pair F1 \u2191")].text ==
                  "0.890000" &&
              overview_row[column_index(overview, "Semantic impurity \u2193")]
                      .text == "0.200000" &&
              overview_row[column_index(overview, "Fragmentation \u2193")]
                      .text == "0.300000" &&
              overview_row[column_index(overview, "Combined quality \u2191")]
                      .text == "0.700000",
          "overview headline values/direction arrows are wrong") ||
      !expect(
          std::holds_alternative<double>(
              *overview_row[column_index(overview, "ARI \u2191")].sort_value),
          "overview metrics must carry typed numeric sort values") ||
      !expect(std::set<std::string>(overview.columns.begin(),
                                    overview.columns.end())
                      .size() == overview.columns.size(),
              "overview columns are not collision-safe")) {
    return 1;
  }

  // Parameters are long-form and run-wide: every effective option, captured
  // payload parameter, and direct experiment parameter occurs exactly once.
  const auto &parameters = snapshot_for(snapshots, "Parameters");
  const auto source_column = column_index(parameters, "Source");
  const auto parameter_column = column_index(parameters, "Parameter");
  const auto value_column = column_index(parameters, "Value");
  const auto &parameter_rows = parameters.frames.back().rows;
  std::set<std::pair<std::string, std::string>> parameter_identities;
  for (const auto &row : parameter_rows) {
    parameter_identities.emplace(row[source_column].text,
                                 row[parameter_column].text);
  }
  if (!expect(!std::ranges::contains(parameters.columns, std::string("Unit")),
              "run-wide Parameters must not repeat per unit") ||
      !expect(parameter_rows.size() == 15 &&
                  parameter_identities.size() == parameter_rows.size(),
              "parameters were repeated, lost, or conflated") ||
      !expect(
          parameter_identities.contains({"Evaluator", "power"}) &&
              parameter_identities.contains(
                  {"Clustering", "labels.cluster.covariance_type"}) &&
              parameter_identities.contains({"Payload", "metric"}) &&
              parameter_identities.contains({"Payload", "evaluation.note"}) &&
              parameter_identities.contains({"Experiment", "dataset"}),
          "readable parameter source classification is wrong")) {
    return 1;
  }
  const auto power_row =
      std::ranges::find_if(parameter_rows, [&](const auto &row) {
        return row[source_column].text == "Evaluator" &&
               row[parameter_column].text == "power";
      });
  const auto names_row =
      std::ranges::find_if(parameter_rows, [&](const auto &row) {
        return row[source_column].text == "Evaluator" &&
               row[parameter_column].text == "dimension_names";
      });
  if (!expect(
          power_row != parameter_rows.end() &&
              (*power_row)[value_column].text == "2",
          "effective evaluator power did not override stale payload data") ||
      !expect(names_row != parameter_rows.end() &&
                  (*names_row)[value_column].text.size() == 512,
              "parameter values are not bounded")) {
    return 1;
  }

  // Every stored MetricValue appears exactly once across the family tabs. The
  // copied combined components remain in Combined, retaining their original
  // descriptor family only in hover metadata.
  if (!expect(
          snapshot_for(snapshots, "Exact").frames.back().rows.size() == 10 &&
              snapshot_for(snapshots, "Semantic").frames.back().rows.size() ==
                  9 &&
              snapshot_for(snapshots, "Fragmentation")
                      .frames.back()
                      .rows.size() == 3 &&
              snapshot_for(snapshots, "Combined").frames.back().rows.size() ==
                  3 &&
              snapshot_for(snapshots, "Alignment").frames.back().rows.size() ==
                  8 &&
              total_metric_rows(snapshots) == 33,
          "stored metrics were dropped or duplicated across family tabs")) {
    return 1;
  }
  std::map<std::string, std::size_t> raw_metric_counts;
  for (const auto label :
       {"Exact", "Semantic", "Fragmentation", "Combined", "Alignment"}) {
    const auto &family = snapshot_for(snapshots, label);
    const auto metric_column = column_index(family, "Metric");
    for (const auto &row : family.frames.back().rows) {
      const auto raw_id = hover_value(row[metric_column], "raw id");
      if (!expect(
              raw_id.has_value() &&
                  hover_value(row[metric_column], "family").has_value() &&
                  hover_value(row[metric_column], "averaging").has_value() &&
                  hover_value(row[metric_column], "direction").has_value(),
              "metric hover lost raw descriptor metadata")) {
        return 1;
      }
      ++raw_metric_counts[*raw_id];
    }
  }
  for (const auto &descriptor_value : ml::clustering_metric_descriptors()) {
    if (!expect(raw_metric_counts.contains(std::string(descriptor_value.name)),
                "stored metric id is missing: " +
                    std::string(descriptor_value.name))) {
      return 1;
    }
  }
  const auto &combined = snapshot_for(snapshots, "Combined");
  const auto combined_metric = column_index(combined, "Metric");
  std::set<std::string> combined_ids;
  for (const auto &row : combined.frames.back().rows) {
    combined_ids.insert(*hover_value(row[combined_metric], "raw id"));
  }
  if (!expect(combined_ids == std::set<std::string>{"combined_quality",
                                                    "fragmentation_micro",
                                                    "semantic_impurity_micro"},
              "Combined tab does not own all three stored combined records")) {
    return 1;
  }

  const auto &exact = snapshot_for(snapshots, "Exact");
  const auto adjusted_rand =
      row_with(exact, "Metric", "Adjusted Rand Index \u2191");
  const auto &semantic = snapshot_for(snapshots, "Semantic");
  const auto severity =
      row_with(semantic, "Metric", "Conditional Merge Error Severity \u2193");
  const auto semantic_value = column_index(semantic, "Value");
  const auto semantic_status = column_index(semantic, "Status");
  if (!expect(adjusted_rand != nullptr,
              "metric names were not humanized with direction arrows") ||
      !expect(severity != nullptr &&
                  (*severity)[semantic_value].text == "N/A" &&
                  !(*severity)[semantic_value].sort_value.has_value() &&
                  hover_value((*severity)[semantic_value], "reason") ==
                      std::optional<std::string>("no_merge_error_pairs") &&
                  (*severity)[semantic_status].text == "No Merge Error Pairs",
              "N/A reason/status/direction presentation is wrong")) {
    return 1;
  }

  // Presentation changes update every owned tab without changing any data.
  table.set_property("group", std::string("compare"));
  table.set_property("title", std::string("GMM comparison"));
  snapshots = view->snapshots_copy();
  if (!expect(std::ranges::all_of(snapshots,
                                  [](const auto &snapshot) {
                                    return snapshot.group == "compare" &&
                                           snapshot.title == "GMM comparison" &&
                                           snapshot.group_layout ==
                                               plot::TableGroupLayout::Tabs;
                                  }),
              "group/title did not update every owned tab") ||
      !expect(total_metric_rows(snapshots) == 33,
              "presentation change altered stored metric rows")) {
    return 1;
  }

  // Accumulate adds comparison rows for Overview/Parameters, but gives each
  // metric family an unbounded run-history frame for the vertical scrubber.
  const auto rows_before_mode_change = total_metric_rows(snapshots);
  table.set_property("update_mode", std::string("accumulate"));
  if (!expect(table.property_as<std::string>("active_update_mode") ==
                      std::optional<std::string>("accumulate") &&
                  total_metric_rows(view->snapshots_copy()) ==
                      rows_before_mode_change,
              "update_mode change replayed data or failed to resolve")) {
    return 1;
  }
  (void)table.process(evaluation_buffer(evaluation_result(), true));
  snapshots = view->snapshots_copy();
  if (!expect(snapshot_for(snapshots, "Overview").frames.back().rows.size() ==
                      2 &&
                  snapshot_for(snapshots, "Overview").frames.size() == 1,
              "accumulate did not retain Overview comparison rows") ||
      !expect(snapshot_for(snapshots, "Parameters").frames.back().rows.size() ==
                      31 &&
                  snapshot_for(snapshots, "Parameters").frames.size() == 1,
              "accumulate repeated or lost long-form parameter rows")) {
    return 1;
  }
  for (const auto label :
       {"Exact", "Semantic", "Fragmentation", "Combined", "Alignment"}) {
    const auto &snapshot = snapshot_for(snapshots, label);
    if (!expect(snapshot.frames.size() == 2 && snapshot.frames.front().n == 1 &&
                    snapshot.frames.back().n == 2,
                "family accumulate did not retain one frame per run for " +
                    std::string(label))) {
      return 1;
    }
  }

  // An accumulated run without optional modes retains prior optional history
  // and adds an empty current frame. Semantic-disabled records remain
  // inspectable.
  (void)table.process(evaluation_buffer(result_without_optional_modes()));
  snapshots = view->snapshots_copy();
  if (!expect(
          snapshot_for(snapshots, "Combined").frames.size() == 3 &&
              snapshot_for(snapshots, "Combined").frames.back().rows.empty() &&
              snapshot_for(snapshots, "Alignment").frames.size() == 3 &&
              snapshot_for(snapshots, "Alignment").frames.back().rows.empty(),
          "accumulate-mode optional history was erased or misrepresented")) {
    return 1;
  }
  const auto &disabled_semantic = snapshot_for(snapshots, "Semantic");
  const auto disabled_value = column_index(disabled_semantic, "Value");
  if (!expect(disabled_semantic.frames.size() == 3 &&
                  !disabled_semantic.frames.back().rows.empty() &&
                  std::ranges::all_of(disabled_semantic.frames.back().rows,
                                      [disabled_value](const auto &row) {
                                        return row[disabled_value].text ==
                                                   "N/A" &&
                                               hover_value(row[disabled_value],
                                                           "reason") ==
                                                   std::optional<std::string>(
                                                       "semantic_disabled");
                                      }),
              "semantic-disabled metrics were omitted instead of preserved")) {
    return 1;
  }

  // Membership comes from the view, not duplicated element state. Once a user
  // clears an optional tab, another accumulated run without that mode must not
  // resurrect an empty tab; other retained optional histories continue
  // normally.
  if (!expect(view->erase("metrics.Combined"),
              "optional tab erase did not find Combined")) {
    return 1;
  }
  (void)table.process(evaluation_buffer(result_without_optional_modes()));
  snapshots = view->snapshots_copy();
  if (!expect(
          !has_snapshot(snapshots, "Combined") &&
              snapshot_for(snapshots, "Alignment").frames.size() == 4 &&
              snapshot_for(snapshots, "Alignment").frames.back().rows.empty(),
          "a cleared optional tab was resurrected by stale element state")) {
    return 1;
  }

  // Replace discards all prior rows/history. Optional tabs disappear when the
  // latest replacement payload no longer contains those stored result modes.
  table.set_property("update_mode", std::string("replace"));
  (void)table.process(evaluation_buffer(result_without_optional_modes()));
  snapshots = view->snapshots_copy();
  if (!expect(snapshots.size() == 5 && !has_snapshot(snapshots, "Combined") &&
                  !has_snapshot(snapshots, "Alignment"),
              "replace did not erase stale optional tabs") ||
      !expect(
          snapshot_for(snapshots, "Overview").frames.size() == 1 &&
              snapshot_for(snapshots, "Overview").frames.back().rows.size() ==
                  1 &&
              snapshot_for(snapshots, "Exact").frames.size() == 1 &&
              snapshot_for(snapshots, "Parameters").frames.size() == 1 &&
              snapshot_for(snapshots, "Parameters").frames.back().rows.size() ==
                  15,
          "replace did not retain only the latest run")) {
    return 1;
  }
  const auto &latest_overview = snapshot_for(snapshots, "Overview");
  const auto &latest_overview_row = latest_overview.frames.back().rows.front();
  const auto combined_headline =
      column_index(latest_overview, "Combined quality \u2191");
  if (!expect(
          latest_overview_row[combined_headline].text == "N/A" &&
              hover_value(latest_overview_row[combined_headline], "reason") ==
                  std::optional<std::string>("not requested"),
          "Overview did not explain absent combined quality")) {
    return 1;
  }

  // Tab order is explicit rather than accidental insertion order. Optional
  // tabs enabled after an exact-only run, and a cleared/recreated Overview,
  // still render in the documented sequence with Overview first.
  (void)table.process(evaluation_buffer());
  if (!expect(view->erase("metrics.Overview"),
              "Overview erase did not find its owned tab")) {
    return 1;
  }
  (void)table.process(evaluation_buffer());
  snapshots = view->snapshots_copy();
  std::ranges::stable_sort(snapshots, {}, &plot::TableSnapshot::tab_order);
  std::vector<std::string> ordered_labels;
  ordered_labels.reserve(snapshots.size());
  for (const auto &snapshot : snapshots) {
    ordered_labels.push_back(snapshot.tab_label);
  }
  if (!expect(ordered_labels == expected_labels,
              "recreated tabs lost the documented Overview-first order")) {
    return 1;
  }

  // Strict sink input validation and normal runtime clearing are unchanged.
  leakflow::Buffer wrong{leakflow::Caps("leakflow/buffer")};
  if (!expect(throws_invalid_argument([&] { (void)table.process(wrong); }),
              "wrong input caps must be rejected") ||
      !expect(
          throws_invalid_argument([&] { (void)table.process(std::nullopt); }),
          "missing input must be rejected")) {
    return 1;
  }
  const auto before_tab_clear = view->snapshots_copy().size();
  if (!expect(view->erase("metrics.Semantic") &&
                  view->snapshots_copy().size() == before_tab_clear - 1 &&
                  has_snapshot(view->snapshots_copy(), "Overview"),
              "per-tab clear removed the wrong owned snapshots")) {
    return 1;
  }
  runtime->clear();
  if (!expect(view->empty() && !runtime->has_sessions(),
              "runtime clear did not clear all table tabs")) {
    return 1;
  }
  return 0;
}
