#include "leakflow/plugins/ml/clustering_evaluate.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/core/property.hpp"
#include "leakflow/ml/clustering_evaluation.hpp"
#include "leakflow/plugins/ml/clustering_evaluation_payload.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::ml {
namespace {

[[nodiscard]] const Buffer &required_input(const ElementInputs &inputs,
                                           std::string_view pad,
                                           std::string_view element) {
  const auto found = inputs.find(std::string(pad));
  if (found == inputs.end() || !found->second) {
    throw std::invalid_argument(std::string(element) +
                                " requires connected input pad " +
                                std::string(pad));
  }
  return *found->second;
}

[[nodiscard]] torch::Tensor tensor_input(const Buffer &buffer,
                                         std::string_view pad) {
  if (buffer.caps().type() != leakflow::base::torch_tensor_caps_type) {
    throw std::invalid_argument(std::string(pad) +
                                " input must have leakflow/torch-tensor caps");
  }
  const auto payload = buffer.payload_as<leakflow::base::TorchTensorPayload>();
  if (!payload) {
    throw std::invalid_argument(std::string(pad) +
                                " input must carry a TorchTensorPayload");
  }
  return payload->tensor();
}

[[nodiscard]] std::string string_property_or(const Element &element,
                                             std::string_view name,
                                             std::string fallback) {
  if (const auto value = element.property_as<std::string>(name)) {
    return *value;
  }
  return fallback;
}

[[nodiscard]] std::int64_t integer_property_or(const Element &element,
                                               std::string_view name,
                                               std::int64_t fallback) {
  if (const auto value = element.property_as<std::int64_t>(name)) {
    return *value;
  }
  return fallback;
}

[[nodiscard]] bool bool_property_or(const Element &element,
                                    std::string_view name, bool fallback) {
  if (const auto value = element.property_as<bool>(name)) {
    return *value;
  }
  return fallback;
}

template <typename T>
[[nodiscard]] T list_property_or(const Element &element,
                                 std::string_view name) {
  if (const auto value = element.property_as<T>(name)) {
    return *value;
  }
  return {};
}

[[nodiscard]] leakflow::ml::ClusteringEvaluationOptions
options_from(const Element &element) {
  leakflow::ml::ClusteringEvaluationOptions options;
  const auto detail = string_property_or(element, "detail", "global");
  options.detail = detail == "full"
                       ? leakflow::ml::ClusteringEvaluationDetail::Full
                       : leakflow::ml::ClusteringEvaluationDetail::Global;
  const auto semantic = string_property_or(element, "semantic", "off");
  options.semantic = semantic == "power"
                         ? leakflow::ml::SemanticEvaluationMode::Power
                         : leakflow::ml::SemanticEvaluationMode::Off;
  options.dimension_names =
      list_property_or<StringList>(element, "dimension_names");
  options.semantic_ranges =
      list_property_or<DoubleList>(element, "semantic_ranges");
  options.semantic_weights =
      list_property_or<DoubleList>(element, "semantic_weights");
  options.power = integer_property_or(element, "power", 2);
  const auto alignment = string_property_or(element, "alignment", "none");
  if (alignment == "exact") {
    options.alignment = leakflow::ml::AlignmentEvaluationMode::Exact;
  } else if (alignment == "semantic") {
    options.alignment = leakflow::ml::AlignmentEvaluationMode::Semantic;
  } else if (alignment == "both") {
    options.alignment = leakflow::ml::AlignmentEvaluationMode::Both;
  } else {
    options.alignment = leakflow::ml::AlignmentEvaluationMode::None;
  }
  options.combined_quality =
      bool_property_or(element, "combined_quality", false);
  return options;
}

[[nodiscard]] std::string join_units(const std::vector<std::int64_t> &units) {
  return Units::of(units).format();
}

struct AlignedEvaluationInputs {
  torch::Tensor labels;
  torch::Tensor truth;
  Units output_units;
};

[[nodiscard]] AlignedEvaluationInputs align_inputs(const Element &element,
                                                   torch::Tensor labels,
                                                   const Buffer &labels_buffer,
                                                   torch::Tensor truth,
                                                   const Buffer &truth_buffer) {
  const auto unbatched = labels.dim() == 1 && truth.dim() == 2;
  const auto batched = labels.dim() == 2 && truth.dim() == 3;
  if (!unbatched && !batched) {
    throw std::invalid_argument(element.identity_for_error() +
                                ": expected labels [N] with truth [N,D], or "
                                "labels [U,N] with truth [U,N,D]");
  }

  const auto labels_units = labels_buffer.units().to_vector();
  const auto truth_units = truth_buffer.units().to_vector();
  if (unbatched) {
    if (labels_units.empty() && truth_units.empty()) {
      return {std::move(labels), std::move(truth), Units::none()};
    }
    if (labels_units.size() != 1 || truth_units.size() != 1 ||
        labels_units != truth_units) {
      throw std::invalid_argument(
          element.identity_for_error() +
          ": unbatched labels and truth require the same singleton unit "
          "identity, or neither may carry units");
    }
    return {std::move(labels), std::move(truth), Units::of(labels_units)};
  }

  const auto unit_count = labels.size(0);
  if (labels_units.empty() && truth_units.empty()) {
    if (truth.size(0) != unit_count) {
      throw std::invalid_argument(element.identity_for_error() +
                                  ": labels and truth unit-axis sizes differ");
    }
    return {std::move(labels), std::move(truth), Units::none()};
  }
  if (labels_units.size() != static_cast<std::size_t>(unit_count) ||
      truth_units.size() != static_cast<std::size_t>(truth.size(0))) {
    throw std::invalid_argument(element.identity_for_error() +
                                ": batched labels and truth must both carry "
                                "one unit identity per leading-axis row");
  }

  const auto alignment = align_labels(labels_units, truth_units);
  if (alignment.shared.empty()) {
    throw std::invalid_argument(element.identity_for_error() +
                                ": labels and truth share no units (labels " +
                                join_units(labels_units) + ", truth " +
                                join_units(truth_units) + ")");
  }
  if (!alignment.identical) {
    auto warning = element.make_log_record(
        log::LogLevel::Warning, "element",
        "labels and truth cover different units; evaluating the shared units");
    warning.fields.emplace("labels_units", join_units(labels_units));
    warning.fields.emplace("truth_units", join_units(truth_units));
    warning.fields.emplace("shared_units", join_units(alignment.shared));
    leakflow::log::write(std::move(warning));
    const auto indexes = torch::TensorOptions().dtype(torch::kInt64);
    labels = labels.index_select(0, torch::tensor(alignment.a_indices, indexes))
                 .contiguous();
    truth = truth.index_select(0, torch::tensor(alignment.b_indices, indexes))
                .contiguous();
  }
  return {std::move(labels), std::move(truth), Units::of(alignment.shared)};
}

[[nodiscard]] std::string
detail_name(leakflow::ml::ClusteringEvaluationDetail value) {
  return value == leakflow::ml::ClusteringEvaluationDetail::Full ? "full"
                                                                 : "global";
}

[[nodiscard]] std::string
semantic_name(leakflow::ml::SemanticEvaluationMode value) {
  return value == leakflow::ml::SemanticEvaluationMode::Power ? "power" : "off";
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
  return "none";
}

void add_bounded_parameter(ClusteringEvaluationPayload::Parameters &parameters,
                           std::string name, std::string value) {
  constexpr std::size_t max_parameter_name = 128;
  constexpr std::size_t max_parameter_value = 512;
  if (name.empty() || name.size() > max_parameter_name) {
    throw std::logic_error(
        "clustering evaluation generated an invalid parameter name");
  }
  if (value.size() > max_parameter_value) {
    value.resize(max_parameter_value);
  }
  parameters.emplace(std::move(name), std::move(value));
}

void copy_label_cluster_metadata(
    ClusteringEvaluationPayload::Parameters &parameters, const Buffer &labels,
    std::size_t &remaining) {
  constexpr std::string_view cluster_prefix = "payload.cluster.";
  for (const auto &[key, value] : labels.metadata()) {
    if (remaining == 0) {
      break;
    }
    if (!key.starts_with(cluster_prefix)) {
      continue;
    }
    auto parameter_name =
        std::string("labels.cluster.") + key.substr(cluster_prefix.size());
    if (parameter_name.size() > 128) {
      continue;
    }
    const auto inserted =
        parameters.emplace(std::move(parameter_name), value.substr(0, 512))
            .second;
    if (inserted) {
      --remaining;
    }
  }
}

[[nodiscard]] ClusteringEvaluationPayload::Parameters
result_parameters(const leakflow::ml::ClusteringEvaluationResult &result,
                  const Buffer &labels) {
  ClusteringEvaluationPayload::Parameters parameters;
  const auto &options = result.effective_options;
  add_bounded_parameter(parameters, "evaluation.detail",
                        detail_name(options.detail));
  add_bounded_parameter(parameters, "evaluation.semantic",
                        semantic_name(options.semantic));
  add_bounded_parameter(
      parameters, "evaluation.dimension_names",
      property_value_to_string(StringList(options.dimension_names)));
  add_bounded_parameter(
      parameters, "evaluation.semantic_ranges",
      property_value_to_string(DoubleList(options.semantic_ranges)));
  add_bounded_parameter(
      parameters, "evaluation.semantic_weights",
      property_value_to_string(DoubleList(options.semantic_weights)));
  add_bounded_parameter(parameters, "evaluation.power",
                        std::to_string(options.power));
  add_bounded_parameter(parameters, "evaluation.alignment",
                        alignment_name(options.alignment));
  add_bounded_parameter(parameters, "evaluation.combined_quality",
                        options.combined_quality ? "true" : "false");
  std::size_t remaining_label_cluster_parameters = 32;
  copy_label_cluster_metadata(parameters, labels,
                              remaining_label_cluster_parameters);
  return parameters;
}

} // namespace

ElementDescriptor ClusteringEvaluate::descriptor() {
  const auto evaluation_output = PropertyEffect{
      .kind = PropertyEffectKind::PayloadOutput,
      .scope = PropertyInvalidationScope::Downstream,
      .output_pads = {"evaluation"},
  };
  return {
      .type_name = "ClusteringEvaluate",
      .klass = "Analyze/Evaluation/Clustering",
      .purpose = "evaluate predicted clusters against exact vector-valued "
                 "semantic truth",
      .input_pads =
          {
              Pad("labels", PadDirection::Input,
                  Caps(leakflow::base::torch_tensor_caps_type)),
              Pad("truth", PadDirection::Input,
                  Caps(leakflow::base::torch_tensor_caps_type),
                  PadPresence::Required, PadMetadataRole::Reference),
          },
      .output_pads =
          {
              Pad("evaluation", PadDirection::Output,
                  Caps(clustering_evaluation_caps_type)),
          },
      .property_specs =
          {
              PropertySpec("detail", std::string("global"),
                           "structured result detail", "",
                           StringEnumConstraint{{"global", "full"}}, "",
                           evaluation_output),
              PropertySpec("semantic", std::string("off"),
                           "semantic evaluation mode", "",
                           StringEnumConstraint{{"off", "power"}}, "",
                           evaluation_output),
              PropertySpec("dimension_names", StringList{},
                           "optional semantic dimension names", "",
                           std::monostate{}, "[name_0,name_1,...]",
                           evaluation_output),
              PropertySpec("semantic_ranges", DoubleList{},
                           "one positive normalization range per semantic "
                           "dimension (required for power mode)",
                           "", std::monostate{}, "[range_0,range_1,...]",
                           evaluation_output),
              PropertySpec("semantic_weights", DoubleList{},
                           "optional non-negative semantic dimension weights "
                           "(empty = equal)",
                           "", std::monostate{}, "[weight_0,weight_1,...]",
                           evaluation_output),
              PropertySpec(
                  "power", std::int64_t{2}, "semantic power cost exponent", "",
                  IntRangeConstraint{1, 2}, "1 or 2", evaluation_output),
              PropertySpec(
                  "alignment", std::string("none"),
                  "optional stored cluster-to-truth alignment", "",
                  StringEnumConstraint{{"none", "exact", "semantic", "both"}},
                  "", evaluation_output),
              PropertySpec("combined_quality", false,
                           "emit the optional harmonic "
                           "semantic-cohesion/group-preservation quality",
                           "", std::monostate{}, "", evaluation_output),
          },
      .keywords = {"clustering", "evaluation", "semantic", "alignment",
                   "metrics", "ml"},
      .metadata_set_by_element =
          {
              make_element_metadata_descriptor(
                  "payload.layout", std::string(),
                  "logical layout of the structured evaluation payload",
                  {"unit/evaluation"}),
              make_element_metadata_descriptor(
                  "payload.clustering_evaluation.schema_version",
                  std::int64_t{}, "numeric result schema version", {"4"}),
              make_element_metadata_descriptor(
                  "payload.clustering_evaluation.semantic", std::string(),
                  "effective semantic evaluation mode", {"off", "power"}),
              make_element_metadata_descriptor(
                  "payload.clustering_evaluation.alignment", std::string(),
                  "effective alignment mode",
                  {"none", "exact", "semantic", "both"}),
          },
  };
}

ClusteringEvaluate::ClusteringEvaluate(std::string name)
    : Element(std::move(name)) {
  configure_from_descriptor(descriptor());
}

std::optional<Buffer> ClusteringEvaluate::process(std::optional<Buffer>) {
  throw std::invalid_argument(
      "ClusteringEvaluate requires named labels and truth inputs");
}

std::optional<Buffer> ClusteringEvaluate::process_inputs(ElementInputs inputs) {
  const auto &labels_buffer =
      required_input(inputs, "labels", "ClusteringEvaluate");
  const auto &truth_buffer =
      required_input(inputs, "truth", "ClusteringEvaluate");
  auto aligned =
      align_inputs(*this, tensor_input(labels_buffer, "labels"), labels_buffer,
                   tensor_input(truth_buffer, "truth"), truth_buffer);
  auto result = leakflow::ml::evaluate_clustering(aligned.labels, aligned.truth,
                                                  options_from(*this));
  auto parameters = result_parameters(result, labels_buffer);
  auto payload = std::make_shared<ClusteringEvaluationPayload>(
      std::move(result), std::move(parameters));

  Buffer output{Caps(clustering_evaluation_caps_type)};
  forward_metadata(*this, inputs, output);
  output.set_units(std::move(aligned.output_units));
  output.set_payload(payload);
  output.set_metadata("payload.layout", payload->layout());
  output.set_metadata("payload.clustering_evaluation.schema_version",
                      std::to_string(payload->result().schema_version));
  output.set_metadata(
      "payload.clustering_evaluation.semantic",
      semantic_name(payload->result().effective_options.semantic));
  output.set_metadata(
      "payload.clustering_evaluation.alignment",
      alignment_name(payload->result().effective_options.alignment));

  auto record =
      make_log_record(log::LogLevel::Debug, "element", "evaluated clustering");
  record.fields.emplace("units",
                        std::to_string(payload->result().units.size()));
  record.fields.emplace("observations",
                        std::to_string(payload->result().observation_count));
  leakflow::log::write(std::move(record));
  return output;
}

} // namespace leakflow::plugins::ml
