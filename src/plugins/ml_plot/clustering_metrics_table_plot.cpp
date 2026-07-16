#include "leakflow/plugins/ml_plot/clustering_metrics_table_plot.hpp"

#include "leakflow/ml/clustering_evaluation.hpp"
#include "leakflow/plugins/ml/clustering_evaluation_payload.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plugins::ml_plot {
namespace {

namespace ml = leakflow::ml;
namespace ml_plugin = leakflow::plugins::ml;
namespace plot = leakflow::plot;

[[nodiscard]] std::string string_property_or(const Element &element,
                                             std::string_view name,
                                             std::string fallback) {
  if (const auto value = element.property_as<std::string>(name)) {
    return *value;
  }
  return fallback;
}

[[nodiscard]] const Buffer &required_input(const ElementInputs &inputs,
                                           std::string_view pad) {
  const auto found = inputs.find(std::string(pad));
  if (found == inputs.end() || !found->second) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot requires connected input pad " +
        std::string(pad));
  }
  return *found->second;
}

[[nodiscard]] std::shared_ptr<ml_plugin::ClusteringEvaluationPayload>
payload_for(const Buffer &input) {
  if (input.caps().type() != ml_plugin::clustering_evaluation_caps_type) {
    throw std::invalid_argument("ClusteringMetricsTablePlot requires "
                                "leakflow/clustering-evaluation input caps");
  }
  auto payload = input.payload_as<ml_plugin::ClusteringEvaluationPayload>();
  if (!payload) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot requires a ClusteringEvaluationPayload");
  }
  return payload;
}

struct MetricRow {
  std::int64_t unit_index = 0;
  std::string scope;
  std::string item;
  ml::MetricValue metric;
};

void append_metric(std::vector<MetricRow> &rows, std::int64_t unit,
                   std::string scope, std::string item,
                   const ml::MetricValue &metric) {
  rows.push_back({unit, std::move(scope), std::move(item), metric});
}

[[nodiscard]] std::string
dimension_item(const ml::ClusteringEvaluationResult &result,
               std::int64_t index) {
  if (index >= 0 && static_cast<std::size_t>(index) <
                        result.effective_options.dimension_names.size()) {
    return result.effective_options
        .dimension_names[static_cast<std::size_t>(index)];
  }
  return "dimension_" + std::to_string(index);
}

[[nodiscard]] std::vector<MetricRow>
metric_rows(const ml::ClusteringEvaluationResult &result) {
  std::vector<MetricRow> rows;
  for (std::size_t unit_index = 0; unit_index < result.units.size();
       ++unit_index) {
    const auto unit = static_cast<std::int64_t>(unit_index);
    const auto &value = result.units[unit_index];
    const auto &exact = value.exact;
    append_metric(rows, unit, "global", "", exact.adjusted_rand_index);
    append_metric(rows, unit, "global", "", exact.adjusted_mutual_information);
    append_metric(rows, unit, "global", "", exact.homogeneity);
    append_metric(rows, unit, "global", "", exact.completeness);
    append_metric(rows, unit, "global", "", exact.v_measure);
    append_metric(rows, unit, "global", "", exact.purity);
    append_metric(rows, unit, "global", "", exact.pair_precision);
    append_metric(rows, unit, "global", "", exact.pair_recall);
    append_metric(rows, unit, "global", "", exact.pair_f1);
    append_metric(rows, unit, "global", "",
                  exact.normalized_mutual_information);

    append_metric(rows, unit, "global", "", value.semantic.micro_impurity);
    append_metric(rows, unit, "global", "", value.semantic.macro_impurity);
    append_metric(rows, unit, "global", "", value.semantic.merge_error_rate);
    append_metric(rows, unit, "global", "",
                  value.semantic.conditional_merge_error_severity);
    for (const auto &dimension : value.semantic.dimensions) {
      const auto item = dimension_item(result, dimension.dimension_index);
      append_metric(rows, unit, "dimension", item, dimension.micro_impurity);
      append_metric(rows, unit, "dimension", item, dimension.macro_impurity);
    }
    if (value.semantic.cluster_details) {
      for (const auto &cluster : *value.semantic.cluster_details) {
        append_metric(rows, unit, "predicted_cluster",
                      std::to_string(cluster.predicted_cluster_index),
                      cluster.impurity);
      }
    }

    append_metric(rows, unit, "global", "", value.fragmentation.micro);
    append_metric(rows, unit, "global", "", value.fragmentation.macro);
    if (value.fragmentation.group_details) {
      for (const auto &group : *value.fragmentation.group_details) {
        append_metric(rows, unit, "truth_group",
                      std::to_string(group.truth_group_index),
                      group.fragmentation);
      }
    }

    if (value.combined_quality) {
      append_metric(rows, unit, "global", "", value.combined_quality->quality);
      append_metric(rows, unit, "combined_component", "semantic_micro_impurity",
                    value.combined_quality->semantic_micro_impurity);
      append_metric(rows, unit, "combined_component", "fragmentation_micro",
                    value.combined_quality->fragmentation_micro);
    }

    if (value.exact_alignment) {
      append_metric(rows, unit, "global", "",
                    value.exact_alignment->matched_accuracy);
      if (value.exact_alignment->truth_group_details) {
        for (const auto &group : *value.exact_alignment->truth_group_details) {
          const auto item = std::to_string(group.truth_group_index);
          append_metric(rows, unit, "truth_group", item, group.precision);
          append_metric(rows, unit, "truth_group", item, group.recall);
          append_metric(rows, unit, "truth_group", item, group.f1);
          append_metric(rows, unit, "truth_group", item, group.jaccard);
        }
      }
    }
    if (value.semantic_alignment) {
      append_metric(rows, unit, "global", "",
                    value.semantic_alignment->normalized_cost);
      for (const auto &dimension : value.semantic_alignment->dimensions) {
        append_metric(rows, unit, "dimension",
                      dimension_item(result, dimension.dimension_index),
                      dimension.normalized_error);
      }
    }
  }
  return rows;
}

[[nodiscard]] std::string_view
detail_name(ml::ClusteringEvaluationDetail detail) {
  switch (detail) {
  case ml::ClusteringEvaluationDetail::Global:
    return "global";
  case ml::ClusteringEvaluationDetail::Full:
    return "full";
  }
  throw std::invalid_argument(
      "ClusteringMetricsTablePlot received an invalid evaluation detail");
}

[[nodiscard]] std::string_view
semantic_name(ml::SemanticEvaluationMode semantic) {
  switch (semantic) {
  case ml::SemanticEvaluationMode::Off:
    return "off";
  case ml::SemanticEvaluationMode::Power:
    return "power";
  }
  throw std::invalid_argument(
      "ClusteringMetricsTablePlot received an invalid semantic mode");
}

[[nodiscard]] std::string_view
alignment_name(ml::AlignmentEvaluationMode alignment) {
  switch (alignment) {
  case ml::AlignmentEvaluationMode::None:
    return "none";
  case ml::AlignmentEvaluationMode::Exact:
    return "exact";
  case ml::AlignmentEvaluationMode::Semantic:
    return "semantic";
  case ml::AlignmentEvaluationMode::Both:
    return "both";
  }
  throw std::invalid_argument(
      "ClusteringMetricsTablePlot received an invalid alignment mode");
}

[[nodiscard]] std::map<std::string, std::string>
table_parameters(const ml_plugin::ClusteringEvaluationPayload &payload,
                 const Buffer &input) {
  std::map<std::string, std::string> parameters;
  for (const auto &[name, value] : payload.parameters()) {
    parameters.emplace("parameter.payload." + name, value);
  }

  // The structured result is authoritative for evaluator settings. Payload
  // parameters are captured presentation context and may be absent or stale;
  // overwrite their evaluation.* columns with the effective options that
  // actually produced the stored metrics.
  const auto &options = payload.result().effective_options;
  constexpr std::string_view effective_prefix = "parameter.payload.evaluation.";
  const auto set_effective_option = [&](std::string_view name,
                                        std::string value) {
    value.resize(std::min<std::size_t>(value.size(), 512));
    parameters[std::string(effective_prefix) + std::string(name)] =
        std::move(value);
  };
  set_effective_option("detail", std::string(detail_name(options.detail)));
  set_effective_option("semantic",
                       std::string(semantic_name(options.semantic)));
  set_effective_option("dimension_names", property_value_to_string(StringList(
                                              options.dimension_names)));
  set_effective_option("semantic_ranges", property_value_to_string(DoubleList(
                                              options.semantic_ranges)));
  set_effective_option("semantic_weights", property_value_to_string(DoubleList(
                                               options.semantic_weights)));
  set_effective_option("power", std::to_string(options.power));
  set_effective_option("alignment",
                       std::string(alignment_name(options.alignment)));
  set_effective_option("combined_quality",
                       options.combined_quality ? "true" : "false");

  constexpr std::string_view metadata_prefix = "payload.parameter.";
  std::size_t remaining_metadata_parameters = 32;
  for (const auto &[key, value] : input.metadata()) {
    if (remaining_metadata_parameters == 0) {
      break;
    }
    if (key.starts_with(metadata_prefix) &&
        key.size() - metadata_prefix.size() <= 118) {
      parameters["parameter.metadata." + key.substr(metadata_prefix.size())] =
          value.substr(0, 512);
      --remaining_metadata_parameters;
    }
  }
  return parameters;
}

[[nodiscard]] plot::TableCell text_cell(std::string value) {
  plot::TableCell cell;
  cell.text = value;
  cell.sort_value = std::move(value);
  return cell;
}

[[nodiscard]] plot::TableCell int_cell(std::int64_t value) {
  plot::TableCell cell;
  cell.text = std::to_string(value);
  cell.sort_value = value;
  return cell;
}

[[nodiscard]] plot::TableCell uint_cell(std::uint64_t value) {
  plot::TableCell cell;
  cell.text = std::to_string(value);
  cell.sort_value = value;
  return cell;
}

[[nodiscard]] plot::TableCell metric_value_cell(const ml::MetricValue &metric) {
  plot::TableCell cell;
  if (metric.value) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << *metric.value;
    cell.text = output.str();
    cell.sort_value = *metric.value;
  } else {
    cell.text = "N/A";
    cell.hover.emplace_back(
        "reason",
        std::string(ml::metric_undefined_reason_name(metric.undefined_reason)));
    cell.hover.emplace_back("support", std::to_string(metric.support_count));
  }
  return cell;
}

[[nodiscard]] std::int64_t unit_identity(const Buffer &input,
                                         std::int64_t dense_unit) {
  if (!input.units().empty() && dense_unit >= 0 &&
      dense_unit < input.units().size()) {
    return input.units().at(dense_unit);
  }
  return dense_unit;
}

std::optional<Buffer> capture_table(Element &element, plot::TableView &view,
                                    const Buffer &input,
                                    std::int64_t &sequence) {
  const auto payload = payload_for(input);
  const auto &result = payload->result();
  const auto parameters = table_parameters(*payload, input);
  const auto metrics = metric_rows(result);
  const auto run = ++sequence;

  plot::TableUpdate update;
  update.group = string_property_or(element, "group", "clustering");
  update.title =
      string_property_or(element, "title", "Clustering evaluation metrics");
  update.update_mode =
      string_property_or(element, "update_mode", "replace") == "append"
          ? plot::TableUpdateMode::AppendRows
          : plot::TableUpdateMode::ReplaceFrame;
  update.max_history = 1;
  update.frame.n = run;
  update.frame.caption = "run = " + std::to_string(run);
  update.columns = {"run", "unit", "observations", "truth_groups",
                    "predicted_clusters"};
  for (const auto &[name, value] : parameters) {
    static_cast<void>(value);
    update.columns.push_back(name);
  }
  update.columns.insert(update.columns.end(),
                        {"scope", "item", "metric", "family", "averaging",
                         "direction", "value", "support", "status"});

  update.frame.rows.reserve(metrics.size());
  for (const auto &row : metrics) {
    const auto &unit =
        result.units.at(static_cast<std::size_t>(row.unit_index));
    const auto &descriptor =
        ml::clustering_metric_descriptor(row.metric.metric);
    std::vector<plot::TableCell> cells;
    cells.reserve(update.columns.size());
    cells.push_back(int_cell(run));
    cells.push_back(int_cell(unit_identity(input, row.unit_index)));
    cells.push_back(int_cell(unit.observation_count));
    cells.push_back(int_cell(unit.truth_group_count));
    cells.push_back(int_cell(unit.predicted_cluster_count));
    for (const auto &[name, value] : parameters) {
      static_cast<void>(name);
      cells.push_back(text_cell(value));
    }
    cells.push_back(text_cell(row.scope));
    cells.push_back(text_cell(row.item));
    cells.push_back(text_cell(std::string(descriptor.name)));
    cells.push_back(
        text_cell(std::string(ml::metric_family_name(descriptor.family))));
    cells.push_back(text_cell(
        std::string(ml::metric_averaging_name(descriptor.averaging))));
    cells.push_back(text_cell(
        std::string(ml::metric_direction_name(descriptor.direction))));
    cells.push_back(metric_value_cell(row.metric));
    cells.push_back(uint_cell(row.metric.support_count));
    cells.push_back(text_cell(
        row.metric.defined() ? std::string("defined")
                             : std::string(ml::metric_undefined_reason_name(
                                   row.metric.undefined_reason))));
    update.frame.rows.push_back(std::move(cells));
  }

  view.push(element.name(), update);
  return std::nullopt;
}

} // namespace

ElementDescriptor ClusteringMetricsTablePlot::descriptor() {
  const auto ui_control = PropertyEffect{
      .kind = PropertyEffectKind::UiControl,
      .scope = PropertyInvalidationScope::ElementUi,
  };
  const auto sink_display = PropertyEffect{
      .kind = PropertyEffectKind::SinkDisplay,
      .scope = PropertyInvalidationScope::ElementUi,
  };
  return {
      .type_name = "ClusteringMetricsTablePlot",
      .klass = "Sink/Plot/Table/ClusteringMetrics",
      .purpose = "show clustering metrics and evaluation/producer parameters "
                 "in a sortable comparison table",
      .input_pads =
          {
              Pad("sink", PadDirection::Input,
                  Caps(ml_plugin::clustering_evaluation_caps_type)),
          },
      .property_specs =
          {
              PropertySpec("group", std::string("clustering"),
                           "table comparison group (window)", "",
                           std::monostate{}, "", ui_control),
              PropertySpec("title",
                           std::string("Clustering evaluation metrics"),
                           "table title", "", std::monostate{}, "", ui_control),
              PropertySpec("update_mode", std::string("replace"),
                           "replace current rows or append each new evaluation "
                           "for comparison",
                           "", StringEnumConstraint{{"replace", "append"}}, "",
                           sink_display),
          },
      .keywords = {"plot", "table", "clustering", "evaluation", "metrics",
                   "parameters", "ml"},
  };
}

ClusteringMetricsTablePlot::ClusteringMetricsTablePlot(std::string name)
    : ClusteringMetricsTablePlot(std::make_shared<plot::TableView>(),
                                 std::move(name)) {}

ClusteringMetricsTablePlot::ClusteringMetricsTablePlot(
    std::shared_ptr<plot::TableView> view, std::string name)
    : Element(std::move(name)), view_(std::move(view)) {
  if (!view_) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot requires a TableView");
  }
  configure_from_descriptor(descriptor());
}

void ClusteringMetricsTablePlot::start() { run_sequence_ = 0; }

std::optional<Buffer>
ClusteringMetricsTablePlot::process(std::optional<Buffer> input) {
  if (!input) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot requires an input buffer");
  }
  return capture_table(*this, *view_, *input, run_sequence_);
}

std::optional<Buffer>
ClusteringMetricsTablePlot::process_inputs(ElementInputs inputs) {
  return capture_table(*this, *view_, required_input(inputs, "sink"),
                       run_sequence_);
}

std::shared_ptr<plot::TableView>
ClusteringMetricsTablePlot::table_view() const {
  return view_;
}

void ClusteringMetricsTablePlot::set_table_view(
    std::shared_ptr<plot::TableView> view) {
  if (!view) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot requires a TableView");
  }
  view_ = std::move(view);
}

void ClusteringMetricsTablePlot::property_changed(std::string_view name) {
  if ((name == "group" || name == "title") && view_) {
    static_cast<void>(view_->update_presentation(
        this->name(), string_property_or(*this, "group", "clustering"),
        string_property_or(*this, "title", "Clustering evaluation metrics")));
  }
}

} // namespace leakflow::plugins::ml_plot
