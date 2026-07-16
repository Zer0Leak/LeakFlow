#include "leakflow/plugins/ml_plot/clustering_metrics_table_plot.hpp"

#include "leakflow/ml/clustering_evaluation.hpp"
#include "leakflow/plugins/ml/clustering_evaluation_payload.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
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

[[nodiscard]] bool resolve_accumulate_for(std::string_view update_mode,
                                          bool live_driven) {
  if (update_mode == "accumulate") {
    return true;
  }
  if (update_mode == "replace") {
    return false;
  }
  if (update_mode == "auto") {
    return live_driven;
  }
  throw std::invalid_argument("ClusteringMetricsTablePlot update_mode must be "
                              "auto, accumulate, or replace");
}

enum class TableSection : std::uint8_t {
  Overview,
  Exact,
  Semantic,
  Fragmentation,
  Combined,
  Alignment,
  Heatmap,
  Parameters,
};

inline constexpr std::array all_sections{
    TableSection::Overview, TableSection::Exact,
    TableSection::Semantic, TableSection::Fragmentation,
    TableSection::Combined, TableSection::Alignment,
    TableSection::Heatmap,  TableSection::Parameters,
};

inline constexpr std::size_t max_heatmap_display_cells_per_frame = 1'000'000;

[[nodiscard]] std::string_view section_label(TableSection section) {
  switch (section) {
  case TableSection::Overview:
    return "Overview";
  case TableSection::Exact:
    return "Exact";
  case TableSection::Semantic:
    return "Semantic";
  case TableSection::Fragmentation:
    return "Fragmentation";
  case TableSection::Combined:
    return "Combined";
  case TableSection::Alignment:
    return "Alignment";
  case TableSection::Heatmap:
    return "Heatmap";
  case TableSection::Parameters:
    return "Parameters";
  }
  throw std::invalid_argument(
      "ClusteringMetricsTablePlot received an invalid table section");
}

[[nodiscard]] std::string snapshot_name(std::string_view element_name,
                                        TableSection section) {
  return std::string(element_name) + "." + std::string(section_label(section));
}

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

[[nodiscard]] std::string bounded(std::string value) {
  value.resize(std::min<std::size_t>(value.size(), 512));
  return value;
}

[[nodiscard]] std::string humanize(std::string_view raw) {
  std::string result;
  result.reserve(raw.size());
  bool capitalize = true;
  for (const auto character : raw) {
    if (character == '_' || character == '.') {
      if (!result.empty() && result.back() != ' ') {
        result.push_back(' ');
      }
      capitalize = true;
      continue;
    }
    auto output = character;
    if (capitalize) {
      output = static_cast<char>(
          std::toupper(static_cast<unsigned char>(character)));
    }
    result.push_back(output);
    capitalize = false;
  }
  return result;
}

[[nodiscard]] std::string readable_metric_name(std::string_view raw) {
  if (raw == "v_measure") {
    return "V-measure";
  }
  return humanize(raw);
}

[[nodiscard]] std::string direction_arrow(ml::MetricDirection direction) {
  return direction == ml::MetricDirection::HigherIsBetter ? "\u2191" : "\u2193";
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

struct ParameterRecord {
  std::string source;
  std::string parameter;
  std::string value;
};

[[nodiscard]] std::vector<ParameterRecord>
parameter_records(const ml_plugin::ClusteringEvaluationPayload &payload,
                  const Buffer &input) {
  const auto &options = payload.result().effective_options;
  std::vector<ParameterRecord> records{
      {"Evaluator", "detail", std::string(detail_name(options.detail))},
      {"Evaluator", "semantic", std::string(semantic_name(options.semantic))},
      {"Evaluator", "dimension_names",
       bounded(property_value_to_string(StringList(options.dimension_names)))},
      {"Evaluator", "semantic_ranges",
       bounded(property_value_to_string(DoubleList(options.semantic_ranges)))},
      {"Evaluator", "semantic_weights",
       bounded(property_value_to_string(DoubleList(options.semantic_weights)))},
      {"Evaluator", "power", std::to_string(options.power)},
      {"Evaluator", "alignment",
       std::string(alignment_name(options.alignment))},
      {"Evaluator", "combined_quality",
       options.combined_quality ? "true" : "false"},
  };

  // Effective options above are authoritative. The payload also carries these
  // exact keys for summaries, so skip only those known duplicates. Unknown
  // evaluation.* keys remain visible as forward-compatible payload context.
  constexpr std::array<std::string_view, 8> effective_parameter_names{
      "evaluation.detail",           "evaluation.semantic",
      "evaluation.dimension_names",  "evaluation.semantic_ranges",
      "evaluation.semantic_weights", "evaluation.power",
      "evaluation.alignment",        "evaluation.combined_quality",
  };
  for (const auto &[name, value] : payload.parameters()) {
    if (!std::ranges::contains(effective_parameter_names, name)) {
      records.push_back(
          {name.starts_with("labels.cluster.") ? "Clustering" : "Payload", name,
           bounded(value)});
    }
  }

  constexpr std::string_view metadata_prefix = "payload.parameter.";
  std::size_t remaining = 32;
  for (const auto &[key, value] : input.metadata()) {
    if (remaining == 0) {
      break;
    }
    if (key.starts_with(metadata_prefix) &&
        key.size() - metadata_prefix.size() <= 118) {
      records.push_back(
          {"Experiment", key.substr(metadata_prefix.size()), bounded(value)});
      --remaining;
    }
  }
  return records;
}

struct OverviewParameter {
  std::string header;
  std::string value;
};

[[nodiscard]] std::optional<std::string>
parameter_value(const std::vector<ParameterRecord> &parameters,
                std::string_view source, std::string_view name) {
  const auto found =
      std::ranges::find_if(parameters, [&](const auto &parameter) {
        return parameter.source == source && parameter.parameter == name;
      });
  return found == parameters.end() ? std::nullopt : std::optional(found->value);
}

[[nodiscard]] std::vector<OverviewParameter>
overview_parameters(const std::vector<ParameterRecord> &parameters) {
  std::vector<OverviewParameter> result;
  std::map<std::string, std::size_t> header_counts;
  for (const auto &parameter : parameters) {
    if (parameter.source == "Clustering") {
      if (parameter.parameter == "labels.cluster.n_features") {
        continue; // promoted to the explicit Features (S) shape column
      }
      auto header =
          "Clustering: " + humanize(parameter.parameter.substr(
                               std::string_view("labels.cluster.").size()));
      ++header_counts[header];
      result.push_back({std::move(header), parameter.value});
    } else if (parameter.source == "Experiment") {
      auto header = "Experiment: " + humanize(parameter.parameter);
      ++header_counts[header];
      result.push_back({std::move(header), parameter.value});
    }
  }

  // Humanization can make different raw spellings look alike. Keep the common
  // case terse, but retain the raw key when disambiguation is needed.
  std::size_t selected_index = 0;
  for (const auto &parameter : parameters) {
    const auto selected =
        (parameter.source == "Clustering" &&
         parameter.parameter != "labels.cluster.n_features") ||
        parameter.source == "Experiment";
    if (!selected) {
      continue;
    }
    auto &entry = result.at(selected_index++);
    if (header_counts.at(entry.header) > 1) {
      entry.header += " [" + parameter.parameter + "]";
    }
  }
  return result;
}

struct MetricRow {
  std::int64_t unit_index = 0;
  TableSection section = TableSection::Exact;
  std::string scope;
  std::string item;
  ml::MetricValue metric;
};

void append_metric(std::vector<MetricRow> &rows, std::int64_t unit,
                   TableSection section, std::string scope, std::string item,
                   const ml::MetricValue &metric) {
  rows.push_back({unit, section, std::move(scope), std::move(item), metric});
}

[[nodiscard]] std::string
dimension_item(const ml::ClusteringEvaluationResult &result,
               std::int64_t index) {
  if (index >= 0 && static_cast<std::size_t>(index) <
                        result.effective_options.dimension_names.size()) {
    return result.effective_options
        .dimension_names[static_cast<std::size_t>(index)];
  }
  return "Dimension " + std::to_string(index);
}

[[nodiscard]] std::vector<MetricRow>
metric_rows(const ml::ClusteringEvaluationResult &result) {
  std::vector<MetricRow> rows;
  for (std::size_t unit_index = 0; unit_index < result.units.size();
       ++unit_index) {
    const auto unit = static_cast<std::int64_t>(unit_index);
    const auto &value = result.units[unit_index];
    const auto &exact = value.exact;
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.adjusted_rand_index);
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.adjusted_mutual_information);
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.homogeneity);
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.completeness);
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.v_measure);
    append_metric(rows, unit, TableSection::Exact, "Global", "", exact.purity);
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.pair_precision);
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.pair_recall);
    append_metric(rows, unit, TableSection::Exact, "Global", "", exact.pair_f1);
    append_metric(rows, unit, TableSection::Exact, "Global", "",
                  exact.normalized_mutual_information);

    append_metric(rows, unit, TableSection::Semantic, "Global", "",
                  value.semantic.micro_impurity);
    append_metric(rows, unit, TableSection::Semantic, "Global", "",
                  value.semantic.macro_impurity);
    append_metric(rows, unit, TableSection::Semantic, "Global", "",
                  value.semantic.merge_error_rate);
    append_metric(rows, unit, TableSection::Semantic, "Global", "",
                  value.semantic.conditional_merge_error_severity);
    for (const auto &dimension : value.semantic.dimensions) {
      const auto item = dimension_item(result, dimension.dimension_index);
      append_metric(rows, unit, TableSection::Semantic, "Dimension", item,
                    dimension.micro_impurity);
      append_metric(rows, unit, TableSection::Semantic, "Dimension", item,
                    dimension.macro_impurity);
    }
    if (value.semantic.cluster_details) {
      for (const auto &cluster : *value.semantic.cluster_details) {
        append_metric(rows, unit, TableSection::Semantic, "Predicted cluster",
                      std::to_string(cluster.predicted_cluster_index),
                      cluster.impurity);
      }
    }

    append_metric(rows, unit, TableSection::Fragmentation, "Global", "",
                  value.fragmentation.micro);
    append_metric(rows, unit, TableSection::Fragmentation, "Global", "",
                  value.fragmentation.macro);
    if (value.fragmentation.group_details) {
      for (const auto &group : *value.fragmentation.group_details) {
        append_metric(rows, unit, TableSection::Fragmentation, "Truth group",
                      std::to_string(group.truth_group_index),
                      group.fragmentation);
      }
    }

    if (value.combined_quality) {
      append_metric(rows, unit, TableSection::Combined, "Global", "",
                    value.combined_quality->quality);
      // These are copied component records stored inside the combined result.
      // Keep them beside the quality score even though their descriptors retain
      // their original semantic/fragmentation family in hover metadata.
      append_metric(rows, unit, TableSection::Combined, "Component",
                    "Semantic micro impurity",
                    value.combined_quality->semantic_micro_impurity);
      append_metric(rows, unit, TableSection::Combined, "Component",
                    "Fragmentation micro",
                    value.combined_quality->fragmentation_micro);
    }

    if (value.exact_alignment) {
      append_metric(rows, unit, TableSection::Alignment, "Global", "",
                    value.exact_alignment->matched_accuracy);
      if (value.exact_alignment->truth_group_details) {
        for (const auto &group : *value.exact_alignment->truth_group_details) {
          const auto item = std::to_string(group.truth_group_index);
          append_metric(rows, unit, TableSection::Alignment, "Truth group",
                        item, group.precision);
          append_metric(rows, unit, TableSection::Alignment, "Truth group",
                        item, group.recall);
          append_metric(rows, unit, TableSection::Alignment, "Truth group",
                        item, group.f1);
          append_metric(rows, unit, TableSection::Alignment, "Truth group",
                        item, group.jaccard);
        }
      }
    }
    if (value.semantic_alignment) {
      append_metric(rows, unit, TableSection::Alignment, "Global", "",
                    value.semantic_alignment->normalized_cost);
      for (const auto &dimension : value.semantic_alignment->dimensions) {
        append_metric(rows, unit, TableSection::Alignment, "Dimension",
                      dimension_item(result, dimension.dimension_index),
                      dimension.normalized_error);
      }
    }
  }
  return rows;
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

[[nodiscard]] plot::TableCell
optional_integer_cell(const std::optional<std::string> &value) {
  if (value) {
    std::int64_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value->data(), value->data() + value->size(), parsed);
    if (error == std::errc{} && end == value->data() + value->size() &&
        parsed > 0) {
      return int_cell(parsed);
    }
  }
  auto unavailable = text_cell("N/A");
  unavailable.sort_value.reset();
  unavailable.hover.emplace_back(
      "reason", value ? "clustering producer reported an invalid input "
                        "feature count: " +
                            *value
                      : "clustering producer did not report input feature "
                        "count");
  return unavailable;
}

void append_metric_hover(plot::TableCell &cell, const ml::MetricValue &metric,
                         bool include_reason = true) {
  const auto &descriptor = ml::clustering_metric_descriptor(metric.metric);
  cell.hover.emplace_back("raw id", std::string(descriptor.name));
  cell.hover.emplace_back(
      "family", std::string(ml::metric_family_name(descriptor.family)));
  cell.hover.emplace_back("averaging", std::string(ml::metric_averaging_name(
                                           descriptor.averaging)));
  cell.hover.emplace_back("direction", std::string(ml::metric_direction_name(
                                           descriptor.direction)));
  if (include_reason && !metric.defined()) {
    cell.hover.emplace_back(
        "reason",
        std::string(ml::metric_undefined_reason_name(metric.undefined_reason)));
  }
}

[[nodiscard]] plot::TableCell metric_name_cell(const ml::MetricValue &metric) {
  const auto &descriptor = ml::clustering_metric_descriptor(metric.metric);
  auto cell = text_cell(readable_metric_name(descriptor.name) + " " +
                        direction_arrow(descriptor.direction));
  append_metric_hover(cell, metric);
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

[[nodiscard]] plot::TableCell
headline_metric_value_cell(const ml::MetricValue &metric) {
  auto cell = metric_value_cell(metric);
  append_metric_hover(cell, metric, false);
  return cell;
}

[[nodiscard]] plot::TableCell status_cell(const ml::MetricValue &metric) {
  auto cell =
      text_cell(metric.defined() ? std::string("Defined")
                                 : humanize(ml::metric_undefined_reason_name(
                                       metric.undefined_reason)));
  append_metric_hover(cell, metric);
  return cell;
}

[[nodiscard]] std::vector<std::int64_t>
unit_identities(const Buffer &input, std::size_t result_unit_count) {
  std::vector<std::int64_t> identities;
  if (input.units().empty()) {
    identities.reserve(result_unit_count);
    for (std::size_t dense_unit = 0; dense_unit < result_unit_count;
         ++dense_unit) {
      identities.push_back(static_cast<std::int64_t>(dense_unit));
    }
  } else {
    identities = input.units().to_vector();
    if (identities.size() != result_unit_count) {
      throw std::invalid_argument(
          "ClusteringMetricsTablePlot requires Buffer.units cardinality to "
          "match evaluation result units");
    }
  }
  for (std::size_t left = 0; left < identities.size(); ++left) {
    if (std::ranges::find(
            identities.begin() + static_cast<std::ptrdiff_t>(left + 1),
            identities.end(), identities[left]) != identities.end()) {
      throw std::invalid_argument(
          "ClusteringMetricsTablePlot requires unique Buffer.units identities");
    }
  }
  return identities;
}

[[nodiscard]] plot::TableRowSelector
unit_row_selector(const Element &element,
                  const std::vector<std::int64_t> &unit_ids) {
  plot::TableRowSelector selector{
      .key = element.name() + ".unit",
      .label = "Unit",
      .column = "Unit",
  };
  selector.values.reserve(unit_ids.size());
  for (const auto unit : unit_ids) {
    selector.values.push_back({
        .value = unit,
        .label = std::to_string(unit),
    });
  }
  return selector;
}

[[nodiscard]] plot::TableUpdate
base_update(const Element &element, TableSection section, std::int64_t run) {
  plot::TableUpdate update;
  update.group = string_property_or(element, "group", "clustering");
  update.group_layout = plot::TableGroupLayout::Tabs;
  update.tab_label = std::string(section_label(section));
  update.tab_order = static_cast<int>(section);
  update.title =
      string_property_or(element, "title", "Clustering evaluation metrics");
  update.frame.n = run;
  update.frame.caption = "Run " + std::to_string(run);
  return update;
}

void push_overview(Element &element, plot::TableView &view,
                   const ml::ClusteringEvaluationResult &result,
                   const std::vector<std::int64_t> &unit_ids,
                   const std::vector<ParameterRecord> &parameters,
                   std::int64_t run, bool accumulate) {
  auto update = base_update(element, TableSection::Overview, run);
  update.row_selector = unit_row_selector(element, unit_ids);
  update.update_mode = accumulate ? plot::TableUpdateMode::AppendRows
                                  : plot::TableUpdateMode::ReplaceFrame;
  update.max_history = 1;
  update.columns = {"Run",          "Unit",         "Observations (N)",
                    "Features (S)", "Truth groups", "Predicted clusters"};
  const auto feature_count =
      parameter_value(parameters, "Clustering", "labels.cluster.n_features");
  const auto selected_parameters = overview_parameters(parameters);
  for (const auto &parameter : selected_parameters) {
    update.columns.push_back(parameter.header);
  }
  update.columns.insert(update.columns.end(),
                        {"ARI \u2191", "AMI \u2191", "Pair F1 \u2191",
                         "Semantic impurity \u2193", "Fragmentation \u2193",
                         "Combined quality \u2191"});

  update.frame.rows.reserve(result.units.size());
  for (std::size_t dense_unit = 0; dense_unit < result.units.size();
       ++dense_unit) {
    const auto &unit = result.units[dense_unit];
    std::vector<plot::TableCell> cells;
    cells.reserve(update.columns.size());
    cells.push_back(int_cell(run));
    cells.push_back(int_cell(unit_ids.at(dense_unit)));
    cells.push_back(int_cell(unit.observation_count));
    cells.push_back(optional_integer_cell(feature_count));
    cells.push_back(int_cell(unit.truth_group_count));
    cells.push_back(int_cell(unit.predicted_cluster_count));
    for (const auto &parameter : selected_parameters) {
      cells.push_back(text_cell(parameter.value));
    }
    cells.push_back(headline_metric_value_cell(unit.exact.adjusted_rand_index));
    cells.push_back(
        headline_metric_value_cell(unit.exact.adjusted_mutual_information));
    cells.push_back(headline_metric_value_cell(unit.exact.pair_f1));
    cells.push_back(headline_metric_value_cell(unit.semantic.micro_impurity));
    cells.push_back(headline_metric_value_cell(unit.fragmentation.micro));
    if (unit.combined_quality) {
      cells.push_back(
          headline_metric_value_cell(unit.combined_quality->quality));
    } else {
      auto unavailable = text_cell("N/A");
      unavailable.sort_value.reset();
      unavailable.hover.emplace_back("reason", "not requested");
      cells.push_back(std::move(unavailable));
    }
    update.frame.rows.push_back(std::move(cells));
  }
  view.push(snapshot_name(element.name(), TableSection::Overview), update);
}

void push_metric_section(Element &element, plot::TableView &view,
                         const std::vector<MetricRow> &metrics,
                         const std::vector<std::int64_t> &unit_ids,
                         TableSection section, std::int64_t run,
                         bool accumulate) {
  auto update = base_update(element, section, run);
  update.row_selector = unit_row_selector(element, unit_ids);
  update.update_mode = accumulate ? plot::TableUpdateMode::AppendFrame
                                  : plot::TableUpdateMode::ReplaceFrame;
  update.max_history = accumulate ? 0 : 1;
  update.columns = {"Run",    "Unit",  "Scope",   "Item",
                    "Metric", "Value", "Support", "Status"};
  for (const auto &row : metrics) {
    if (row.section != section) {
      continue;
    }
    std::vector<plot::TableCell> cells;
    cells.reserve(update.columns.size());
    cells.push_back(int_cell(run));
    cells.push_back(
        int_cell(unit_ids.at(static_cast<std::size_t>(row.unit_index))));
    cells.push_back(text_cell(row.scope));
    cells.push_back(text_cell(row.item));
    cells.push_back(metric_name_cell(row.metric));
    cells.push_back(metric_value_cell(row.metric));
    cells.push_back(uint_cell(row.metric.support_count));
    cells.push_back(status_cell(row.metric));
    update.frame.rows.push_back(std::move(cells));
  }
  view.push(snapshot_name(element.name(), section), update);
}

[[nodiscard]] std::string compact_number(const torch::Tensor &value) {
  std::ostringstream output;
  if (value.scalar_type() == c10::ScalarType::UInt64) {
    output << value.item().toUInt64();
  } else if (c10::isIntegralType(value.scalar_type(), /*includeBool=*/true)) {
    output << value.item<std::int64_t>();
  } else {
    output << std::setprecision(6) << value.item<double>();
  }
  return output.str();
}

[[nodiscard]] std::vector<std::string>
truth_group_labels(const torch::Tensor &truth_vectors,
                   const ml::ClusteringEvaluationOptions &options,
                   std::int64_t rows) {
  if (!truth_vectors.defined() || truth_vectors.layout() != torch::kStrided ||
      !truth_vectors.device().is_cpu() ||
      (!c10::isIntegralType(truth_vectors.scalar_type(),
                            /*includeBool=*/true) &&
       !c10::isFloatingType(truth_vectors.scalar_type()))) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap truth-vector tensor contract is "
        "invalid");
  }
  auto values = truth_vectors.contiguous();
  if (values.dim() != 2 || values.size(0) != rows || values.size(1) <= 0) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap truth-vector shape is invalid");
  }
  if (c10::isFloatingType(values.scalar_type()) &&
      !torch::isfinite(values).all().item<bool>()) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap truth vectors must be finite");
  }
  std::vector<std::string> labels;
  labels.reserve(static_cast<std::size_t>(rows));
  for (std::int64_t row = 0; row < rows; ++row) {
    std::ostringstream label;
    label << '(';
    for (std::int64_t dimension = 0; dimension < values.size(1); ++dimension) {
      if (dimension != 0) {
        label << ", ";
      }
      const auto dense_dimension = static_cast<std::size_t>(dimension);
      label << (dense_dimension < options.dimension_names.size() &&
                        !options.dimension_names[dense_dimension].empty()
                    ? options.dimension_names[dense_dimension]
                    : "d" + std::to_string(dimension))
            << '=' << compact_number(values.index({row, dimension}));
    }
    label << ')';
    labels.push_back(label.str());
  }
  return labels;
}

struct HeatmapShape {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::size_t row_count = 0;
  std::size_t col_count = 0;
};

[[nodiscard]] HeatmapShape
heatmap_shape(const ml::ClusteringEvaluationUnitResult &unit) {
  const auto &contingency = unit.partition_detail->contingency;
  const auto rows = contingency.truth_group_count;
  const auto cols = contingency.predicted_cluster_count;
  if (rows <= 0 || cols <= 0 || unit.observation_count <= 0 ||
      rows != unit.truth_group_count || cols != unit.predicted_cluster_count) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap contingency shape is invalid");
  }
  const auto rows_u = static_cast<std::uint64_t>(rows);
  const auto cols_u = static_cast<std::uint64_t>(cols);
  if (rows_u > std::numeric_limits<std::size_t>::max() ||
      cols_u > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap contingency shape is too large");
  }
  return {
      .rows = rows,
      .cols = cols,
      .row_count = static_cast<std::size_t>(rows_u),
      .col_count = static_cast<std::size_t>(cols_u),
  };
}

[[nodiscard]] bool heatmap_frame_exceeds_display_limit(
    const ml::ClusteringEvaluationResult &result) {
  std::size_t total_cells = 0;
  for (const auto &unit : result.units) {
    if (!unit.partition_detail) {
      continue;
    }
    const auto shape = heatmap_shape(unit);
    if (shape.row_count >
        max_heatmap_display_cells_per_frame / shape.col_count) {
      return true;
    }
    const auto cells = shape.row_count * shape.col_count;
    if (cells > max_heatmap_display_cells_per_frame - total_cells) {
      return true;
    }
    total_cells += cells;
  }
  return false;
}

void require_cpu_int64_vector(const torch::Tensor &tensor,
                              std::string_view field) {
  if (!tensor.defined() || tensor.layout() != torch::kStrided ||
      !tensor.device().is_cpu() || tensor.scalar_type() != torch::kInt64 ||
      tensor.dim() != 1) {
    throw std::invalid_argument("ClusteringMetricsTablePlot heatmap " +
                                std::string(field) +
                                " must be a CPU strided int64 vector");
  }
}

[[nodiscard]] plot::TableHeatmapPage
heatmap_page(const ml::ClusteringEvaluationUnitResult &unit,
             const ml::ClusteringEvaluationOptions &options,
             std::int64_t unit_id, std::int64_t run,
             bool frame_exceeds_display_limit) {
  plot::TableHeatmapPage page;
  page.selector_value = unit_id;
  page.row_axis_label = "true group";
  page.col_axis_label = "predicted cluster";
  if (!unit.partition_detail) {
    page.unavailable_reason = "requires ClusteringEvaluate(detail=full)";
    return page;
  }

  const auto shape = heatmap_shape(unit);
  if (frame_exceeds_display_limit) {
    page.unavailable_reason =
        "combined unit contingencies exceed the 1,000,000-cell per-frame "
        "display limit";
    return page;
  }

  const auto &detail = *unit.partition_detail;
  const auto rows = shape.rows;
  const auto cols = shape.cols;
  const auto row_count = shape.row_count;
  const auto col_count = shape.col_count;

  require_cpu_int64_vector(detail.predicted_ids, "predicted_ids");
  auto predicted_ids = detail.predicted_ids.contiguous();
  if (predicted_ids.numel() != cols) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap predicted-id shape is invalid");
  }
  const auto *predicted_id_data = predicted_ids.data_ptr<std::int64_t>();
  for (std::int64_t index = 1; index < cols; ++index) {
    if (predicted_id_data[index] <= predicted_id_data[index - 1]) {
      throw std::invalid_argument(
          "ClusteringMetricsTablePlot heatmap predicted IDs must be strictly "
          "increasing");
    }
  }
  std::vector<std::int64_t> column_order(col_count);
  std::iota(column_order.begin(), column_order.end(), std::int64_t{0});
  auto exact_aligned = false;
  if (unit.exact_alignment) {
    if (unit.exact_alignment->aligned_column_to_predicted_cluster.size() !=
        col_count) {
      throw std::invalid_argument(
          "ClusteringMetricsTablePlot exact heatmap permutation has the wrong "
          "length");
    }
    column_order = unit.exact_alignment->aligned_column_to_predicted_cluster;
    std::vector<bool> seen(col_count, false);
    for (const auto canonical_column : column_order) {
      if (canonical_column < 0 || canonical_column >= cols ||
          seen[static_cast<std::size_t>(canonical_column)]) {
        throw std::invalid_argument(
            "ClusteringMetricsTablePlot exact heatmap permutation is invalid");
      }
      seen[static_cast<std::size_t>(canonical_column)] = true;
    }
    exact_aligned = true;
  }
  std::vector<std::size_t> displayed_column(col_count);
  for (std::size_t column = 0; column < col_count; ++column) {
    displayed_column[static_cast<std::size_t>(column_order[column])] = column;
  }

  require_cpu_int64_vector(detail.contingency.truth_group_indices,
                           "truth_group_indices");
  require_cpu_int64_vector(detail.contingency.predicted_cluster_indices,
                           "predicted_cluster_indices");
  require_cpu_int64_vector(detail.contingency.counts, "counts");
  auto truth_indices = detail.contingency.truth_group_indices.contiguous();
  auto predicted_indices =
      detail.contingency.predicted_cluster_indices.contiguous();
  auto counts = detail.contingency.counts.contiguous();
  if (truth_indices.numel() != predicted_indices.numel() ||
      truth_indices.numel() != counts.numel()) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap sparse arrays are invalid");
  }

  page.rows = rows;
  page.cols = cols;
  page.data.assign(row_count * col_count, 0.0);
  const auto *truth_data = truth_indices.data_ptr<std::int64_t>();
  const auto *predicted_data = predicted_indices.data_ptr<std::int64_t>();
  const auto *count_data = counts.data_ptr<std::int64_t>();
  std::vector<std::uint64_t> truth_support(row_count, 0);
  std::vector<bool> predicted_has_support(col_count, false);
  std::uint64_t total_support = 0;
  std::int64_t previous_truth = -1;
  std::int64_t previous_predicted = -1;
  for (std::int64_t index = 0; index < counts.numel(); ++index) {
    const auto truth_group = truth_data[index];
    const auto predicted_cluster = predicted_data[index];
    if (truth_group < 0 || truth_group >= rows || predicted_cluster < 0 ||
        predicted_cluster >= cols || count_data[index] <= 0) {
      throw std::invalid_argument(
          "ClusteringMetricsTablePlot heatmap sparse entry is invalid");
    }
    if (truth_group < previous_truth ||
        (truth_group == previous_truth &&
         predicted_cluster <= previous_predicted)) {
      throw std::invalid_argument(
          "ClusteringMetricsTablePlot heatmap sparse coordinates must be "
          "strictly ordered and unique");
    }
    previous_truth = truth_group;
    previous_predicted = predicted_cluster;
    const auto count = static_cast<std::uint64_t>(count_data[index]);
    if (count > std::numeric_limits<std::uint64_t>::max() - total_support) {
      throw std::invalid_argument(
          "ClusteringMetricsTablePlot heatmap contingency support overflows");
    }
    total_support += count;
    auto &row_support = truth_support[static_cast<std::size_t>(truth_group)];
    row_support += count;
    predicted_has_support[static_cast<std::size_t>(predicted_cluster)] = true;
    const auto destination =
        static_cast<std::size_t>(truth_group) * col_count +
        displayed_column[static_cast<std::size_t>(predicted_cluster)];
    page.data[destination] = static_cast<double>(count);
  }
  if (total_support != static_cast<std::uint64_t>(unit.observation_count) ||
      std::ranges::any_of(truth_support,
                          [](const auto support) { return support == 0; }) ||
      std::ranges::any_of(predicted_has_support,
                          [](const auto supported) { return !supported; })) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot heatmap contingency support is "
        "inconsistent");
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    const auto begin =
        page.data.begin() + static_cast<std::ptrdiff_t>(row * col_count);
    const auto end = begin + static_cast<std::ptrdiff_t>(col_count);
    const auto support = static_cast<double>(truth_support[row]);
    std::transform(begin, end, begin,
                   [support](double value) { return value / support; });
  }

  page.row_labels = truth_group_labels(detail.truth_vectors, options, rows);
  page.col_labels.reserve(col_count);
  for (const auto canonical_column : column_order) {
    page.col_labels.push_back(std::to_string(
        predicted_id_data[static_cast<std::size_t>(canonical_column)]));
  }
  page.caption = "Run " + std::to_string(run) + " - " +
                 (exact_aligned ? "exact-overlap aligned contingency"
                                : "raw contingency") +
                 " - row-normalized";
  page.vmin = 0.0;
  page.vmax = 1.0;
  return page;
}

void push_heatmap(Element &element, plot::TableView &view,
                  const ml::ClusteringEvaluationResult &result,
                  const std::vector<std::int64_t> &unit_ids, std::int64_t run,
                  bool accumulate) {
  auto update = base_update(element, TableSection::Heatmap, run);
  update.row_selector = unit_row_selector(element, unit_ids);
  update.update_mode = accumulate ? plot::TableUpdateMode::AppendFrame
                                  : plot::TableUpdateMode::ReplaceFrame;
  update.max_history = accumulate ? 0 : 1;
  update.columns = {"Unit"};
  update.frame.heatmap = plot::TableHeatmapFrame{};
  update.frame.rows.reserve(result.units.size());
  update.frame.heatmap->pages.reserve(result.units.size());
  const auto frame_exceeds_display_limit =
      heatmap_frame_exceeds_display_limit(result);
  for (std::size_t dense_unit = 0; dense_unit < result.units.size();
       ++dense_unit) {
    const auto unit_id = unit_ids.at(dense_unit);
    update.frame.rows.push_back({int_cell(unit_id)});
    update.frame.heatmap->pages.push_back(
        heatmap_page(result.units[dense_unit], result.effective_options,
                     unit_id, run, frame_exceeds_display_limit));
  }
  view.push(snapshot_name(element.name(), TableSection::Heatmap), update);
}

void push_parameters(Element &element, plot::TableView &view,
                     const std::vector<ParameterRecord> &parameters,
                     std::int64_t run, bool accumulate) {
  auto update = base_update(element, TableSection::Parameters, run);
  update.update_mode = accumulate ? plot::TableUpdateMode::AppendRows
                                  : plot::TableUpdateMode::ReplaceFrame;
  update.max_history = 1;
  update.columns = {"Run", "Source", "Parameter", "Value"};
  update.frame.rows.reserve(parameters.size());
  for (const auto &parameter : parameters) {
    update.frame.rows.push_back({int_cell(run), text_cell(parameter.source),
                                 text_cell(parameter.parameter),
                                 text_cell(parameter.value)});
  }
  view.push(snapshot_name(element.name(), TableSection::Parameters), update);
}

[[nodiscard]] bool has_section(const std::vector<MetricRow> &metrics,
                               TableSection section) {
  return std::ranges::any_of(
      metrics, [section](const auto &row) { return row.section == section; });
}

std::optional<Buffer> capture_table(Element &element, plot::TableView &view,
                                    const Buffer &input, std::int64_t &sequence,
                                    bool accumulate) {
  const auto payload = payload_for(input);
  const auto &result = payload->result();
  const auto parameters = parameter_records(*payload, input);
  const auto metrics = metric_rows(result);
  const auto unit_ids = unit_identities(input, result.units.size());
  const auto run = ++sequence;

  // Explicit section order keeps Overview first even after a tab is cleared and
  // recreated. Family tabs keep every stored MetricValue in exactly one
  // structural section.
  push_overview(element, view, result, unit_ids, parameters, run, accumulate);
  push_metric_section(element, view, metrics, unit_ids, TableSection::Exact,
                      run, accumulate);
  push_metric_section(element, view, metrics, unit_ids, TableSection::Semantic,
                      run, accumulate);
  push_metric_section(element, view, metrics, unit_ids,
                      TableSection::Fragmentation, run, accumulate);

  if (has_section(metrics, TableSection::Combined)) {
    push_metric_section(element, view, metrics, unit_ids,
                        TableSection::Combined, run, accumulate);
  } else if (accumulate && view.contains(snapshot_name(
                               element.name(), TableSection::Combined))) {
    push_metric_section(element, view, metrics, unit_ids,
                        TableSection::Combined, run, true);
  } else if (!accumulate) {
    static_cast<void>(
        view.erase(snapshot_name(element.name(), TableSection::Combined)));
  }
  if (has_section(metrics, TableSection::Alignment)) {
    push_metric_section(element, view, metrics, unit_ids,
                        TableSection::Alignment, run, accumulate);
  } else if (accumulate && view.contains(snapshot_name(
                               element.name(), TableSection::Alignment))) {
    push_metric_section(element, view, metrics, unit_ids,
                        TableSection::Alignment, run, true);
  } else if (!accumulate) {
    static_cast<void>(
        view.erase(snapshot_name(element.name(), TableSection::Alignment)));
  }
  push_heatmap(element, view, result, unit_ids, run, accumulate);
  push_parameters(element, view, parameters, run, accumulate);
  return std::nullopt;
}

} // namespace

ElementDescriptor ClusteringMetricsTablePlot::descriptor() {
  const auto ui_control = PropertyEffect{
      .kind = PropertyEffectKind::UiControl,
      .scope = PropertyInvalidationScope::ElementUi,
  };
  return {
      .type_name = "ClusteringMetricsTablePlot",
      .klass = "Sink/Plot/Table/ClusteringMetrics",
      .purpose = "show clustering evaluation as readable overview, metric-"
                 "family, heatmap, and parameter tabs",
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
              PropertySpec(
                  "update_mode", std::string("auto"),
                  "how successive buffers combine: accumulate appends each "
                  "buffer as a new trace row (scrub with the slider), replace "
                  "shows only the latest, auto follows liveness",
                  "", StringEnumConstraint{{"auto", "accumulate", "replace"}},
                  "", ui_control),
              PropertySpec(
                  "active_update_mode", std::string("replace"),
                  "resolved update mode currently selected by update_mode", "",
                  StringEnumConstraint{{"accumulate", "replace"}}, "",
                  PropertyEffect{}, /*optional=*/false,
                  /*writable=*/false),
          },
      .keywords = {"plot", "table", "clustering", "evaluation", "metrics",
                   "heatmap", "parameters", "ml"},
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

void ClusteringMetricsTablePlot::start() {
  run_sequence_ = 0;
  update_active_update_mode();
}

bool ClusteringMetricsTablePlot::resolve_accumulate() {
  const auto accumulate = resolve_accumulate_for(
      string_property_or(*this, "update_mode", "auto"), is_live_driven());
  set_read_only_property("active_update_mode",
                         std::string(accumulate ? "accumulate" : "replace"));
  return accumulate;
}

void ClusteringMetricsTablePlot::update_active_update_mode() {
  static_cast<void>(resolve_accumulate());
}

std::optional<Buffer>
ClusteringMetricsTablePlot::process(std::optional<Buffer> input) {
  if (!input) {
    throw std::invalid_argument(
        "ClusteringMetricsTablePlot requires an input buffer");
  }
  return capture_table(*this, *view_, *input, run_sequence_,
                       resolve_accumulate());
}

std::optional<Buffer>
ClusteringMetricsTablePlot::process_inputs(ElementInputs inputs) {
  return capture_table(*this, *view_, required_input(inputs, "sink"),
                       run_sequence_, resolve_accumulate());
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
  if (name == "update_mode") {
    update_active_update_mode();
  }
  if ((name == "group" || name == "title") && view_) {
    const auto group = string_property_or(*this, "group", "clustering");
    const auto title =
        string_property_or(*this, "title", "Clustering evaluation metrics");
    for (const auto section : all_sections) {
      static_cast<void>(view_->update_presentation(
          snapshot_name(this->name(), section), group, title));
    }
  }
}

void ClusteringMetricsTablePlot::live_driven_changed() {
  update_active_update_mode();
}

} // namespace leakflow::plugins::ml_plot
