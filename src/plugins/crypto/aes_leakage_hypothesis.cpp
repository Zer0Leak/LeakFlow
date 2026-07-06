#include "leakflow/plugins/crypto/aes_leakage_hypothesis.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::crypto {
namespace {

inline constexpr auto aes_state_bytes = std::int64_t{16};
inline constexpr auto aes_guess_domain_size = std::int64_t{256};

using AesLeakageChannel = leakflow::crypto::aes::FirstRoundLeakageChannel;

[[nodiscard]] Caps torch_tensor_caps(Caps::Params params = {}) {
  return Caps(leakflow::base::torch_tensor_caps_type, std::move(params));
}

[[nodiscard]] IntList all_byte_indexes() {
  IntList indexes;
  indexes.reserve(static_cast<std::size_t>(aes_state_bytes));
  for (std::int64_t index = 0; index < aes_state_bytes; ++index) {
    indexes.push_back(index);
  }
  return indexes;
}

[[nodiscard]] std::vector<std::size_t>
selected_byte_indexes(const Element &element) {
  auto property_indexes =
      element.property_as<IntList>("byte_indexes").value_or(all_byte_indexes());
  if (property_indexes.empty()) {
    property_indexes = all_byte_indexes();
  }

  std::set<std::int64_t> seen;
  std::vector<std::size_t> indexes;
  indexes.reserve(property_indexes.size());

  for (const auto index : property_indexes) {
    if (index < 0 || index >= aes_state_bytes) {
      throw std::invalid_argument(
          "AesLeakageHypothesis byte_indexes values must be in [0,15]");
    }
    if (!seen.insert(index).second) {
      throw std::invalid_argument(
          "AesLeakageHypothesis byte_indexes must not contain duplicates");
    }
    indexes.push_back(static_cast<std::size_t>(index));
  }

  return indexes;
}

[[nodiscard]] std::string
size_indexes_metadata(const std::vector<std::size_t> &values) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
  return output.str();
}

[[nodiscard]] StringList string_list_property_or(const Element &element,
                                                 std::string_view name,
                                                 StringList fallback) {
  if (const auto value = element.property_as<StringList>(name)) {
    return *value;
  }

  return fallback;
}

[[nodiscard]] std::vector<AesLeakageChannel>
leakage_channels_for(const Element &element) {
  auto values = string_list_property_or(element, "channels",
                                        StringList{aes_leakage_channel_hw_y});
  return leakflow::crypto::aes::parse_first_round_leakage_channels(
      std::span<const std::string>(values.data(), values.size()));
}

[[nodiscard]] std::string
channels_metadata(const std::vector<AesLeakageChannel> &channels) {
  return leakflow::crypto::aes::first_round_leakage_channels_metadata(
      std::span<const AesLeakageChannel>(channels.data(), channels.size()));
}

[[nodiscard]] std::string
channel_dependencies_metadata(const std::vector<AesLeakageChannel> &channels) {
  return leakflow::crypto::aes::
      first_round_leakage_channel_dependencies_metadata(
          std::span<const AesLeakageChannel>(channels.data(),
                                             channels.size()));
}

[[nodiscard]] std::string guess_values_metadata(const IntList &values) {
  if (values.empty()) {
    return "0..255";
  }

  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
  return output.str();
}

[[nodiscard]] torch::Tensor guess_values_tensor_for(const Element &element,
                                                    const torch::Device &device) {
  const auto property_values =
      element.property_as<IntList>("guess_values").value_or(IntList{});
  if (property_values.empty()) {
    return torch::arange(
               0, aes_guess_domain_size,
               torch::TensorOptions().dtype(torch::kLong).device(device))
        .to(torch::kUInt8);
  }

  std::set<std::int64_t> seen;
  std::vector<leakflow::crypto::Byte> values;
  values.reserve(property_values.size());
  for (const auto value : property_values) {
    if (value < 0 || value > 255) {
      throw std::invalid_argument(
          "AesLeakageHypothesis guess_values values must be in [0,255]");
    }
    if (!seen.insert(value).second) {
      throw std::invalid_argument(
          "AesLeakageHypothesis guess_values must not contain duplicates");
    }
    values.push_back(static_cast<leakflow::crypto::Byte>(value));
  }

  return torch::tensor(
             values,
             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
      .to(device);
}

[[nodiscard]] const Buffer &required_input(const ElementInputs &inputs,
                                           std::string_view pad_name,
                                           std::string_view element_name) {
  const auto found = inputs.find(std::string(pad_name));
  if (found == inputs.end() || !found->second) {
    throw std::invalid_argument(std::string(element_name) +
                                " requires connected input pad " +
                                std::string(pad_name));
  }

  return *found->second;
}

[[nodiscard]] std::shared_ptr<leakflow::base::TorchTensorPayload>
torch_payload_for(const Buffer &buffer, std::string_view pad_name) {
  if (buffer.caps().type() != leakflow::base::torch_tensor_caps_type) {
    throw std::invalid_argument(std::string(pad_name) +
                                " input must have leakflow/torch-tensor caps");
  }

  auto payload = buffer.payload_as<leakflow::base::TorchTensorPayload>();
  if (!payload) {
    throw std::invalid_argument(std::string(pad_name) +
                                " input must carry a TorchTensorPayload");
  }

  return payload;
}

void require_uint8_strided(const torch::Tensor &tensor, std::string_view name) {
  if (!tensor.defined()) {
    throw std::invalid_argument(std::string(name) + " tensor must be defined");
  }
  if (tensor.scalar_type() != torch::kUInt8) {
    throw std::invalid_argument(std::string(name) +
                                " tensor must have dtype uint8");
  }
  if (tensor.layout() != torch::kStrided) {
    throw std::invalid_argument(std::string(name) +
                                " tensor must use strided layout");
  }
}

void require_plaintext_blocks(const torch::Tensor &plaintexts) {
  require_uint8_strided(plaintexts, "plaintexts");
  if (plaintexts.dim() != 2 || plaintexts.size(1) != aes_state_bytes) {
    throw std::invalid_argument("plaintexts tensor must have shape [N,16]");
  }
}

[[nodiscard]] std::optional<Buffer>
process_plaintexts(AesLeakageHypothesis &element, const Buffer &input,
                   ElementInputs metadata_inputs) {
  const auto plaintext_payload = torch_payload_for(input, "plaintexts");
  const auto &plaintexts = plaintext_payload->tensor();
  require_plaintext_blocks(plaintexts);

  const auto byte_indexes = selected_byte_indexes(element);
  const auto channels = leakage_channels_for(element);
  const auto guess_values = guess_values_tensor_for(element, plaintexts.device());
  const auto property_guess_values =
      element.property_as<IntList>("guess_values").value_or(IntList{});
  const auto channel_span =
      std::span<const AesLeakageChannel>(channels.data(), channels.size());
  auto hypotheses = leakflow::crypto::aes::first_round_leakage_hypotheses_at(
      guess_values, plaintexts,
      std::span<const std::size_t>(byte_indexes.data(), byte_indexes.size()),
      channel_span);

  auto payload = leakflow::base::TorchTensorPayload(std::move(hypotheses));
  Buffer output{payload.caps()};
  forward_metadata(metadata_inputs, profile_for_klass(element.element_kclass()),
                   output, element.name());
  output.set_metadata("routing.element", element.name());
  output.set_metadata("payload.leakage.model", aes_leakage_model_id);
  output.set_metadata("payload.leakage.hypothesis",
                      aes_leakage_hypothesis_id);
  output.set_metadata("payload.leakage.byte_indexes",
                      size_indexes_metadata(byte_indexes));
  output.set_metadata("payload.leakage.channels", channels_metadata(channels));
  output.set_metadata("payload.crypto.algorithm", "AES");
  output.set_metadata("payload.crypto.state_bytes",
                      std::to_string(aes_state_bytes));
  output.set_metadata("payload.trace.count",
                      std::to_string(plaintexts.size(0)));
  output.set_metadata("attack.hypothesis.algorithm", "aes");
  output.set_metadata("attack.hypothesis.round", "first");
  output.set_metadata("attack.unit.kind", "byte");
  output.set_metadata("attack.unit.indexes", size_indexes_metadata(byte_indexes));
  output.set_metadata("attack.guess.kind", "byte");
  output.set_metadata("attack.guess.count", std::to_string(guess_values.size(0)));
  output.set_metadata("attack.guess.order", "domain");
  output.set_metadata("attack.guess.values",
                      guess_values_metadata(property_guess_values));
  output.set_metadata("attack.channel.depends_on_guess",
                      channel_dependencies_metadata(channels));
  output.set_metadata("tensor.axes",
                      "attack_unit,guess,trace,leakage_channel");
  output.set_payload(
      std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));

  auto record = element.make_log_record(
      log::LogLevel::Debug, "element", "computed AES leakage hypotheses");
  record.fields.emplace("payload.leakage.model", aes_leakage_model_id);
  record.fields.emplace("payload.leakage.channels",
                        output.metadata("payload.leakage.channels"));
  record.fields.emplace("byte_indexes",
                        output.metadata("payload.leakage.byte_indexes"));
  record.fields.emplace("attack.guess.count",
                        output.metadata("attack.guess.count"));
  record.fields.emplace("payload.trace.count",
                        output.metadata("payload.trace.count"));
  leakflow::log::write(std::move(record));

  return output;
}

} // namespace

ElementDescriptor AesLeakageHypothesis::descriptor() {
  return {
      .type_name = "AesLeakageHypothesis",
      .klass = "Analyze/SCA/Hypothesis/AES",
      .purpose = "compute AES predicted leakage for every selected key guess",
      .input_pads =
          {
              Pad("plaintexts", PadDirection::Input,
                  torch_tensor_caps({
                      {leakflow::base::caps_param_dtype, "uint8"},
                      {leakflow::base::caps_param_rank, "2"},
                  })),
          },
      .output_pads =
          {
              Pad("hypotheses", PadDirection::Output,
                  torch_tensor_caps({
                      {leakflow::base::caps_param_dtype, "uint8"},
                      {leakflow::base::caps_param_rank, "4"},
                  })),
          },
      .property_specs =
          {
              PropertySpec(
                  "byte_indexes", all_byte_indexes(),
                  "AES state byte indexes to attack; [] means all bytes",
                  "", std::monostate{}, "",
                  PropertyEffect{
                      .kind = PropertyEffectKind::PayloadOutput,
                      .scope = PropertyInvalidationScope::Downstream,
                      .output_pads = {"hypotheses"},
                  }),
              PropertySpec(
                  "channels",
                  StringList{aes_leakage_channel_hw_y},
                  "predicted leakage channels to output; order controls axis 3",
                  "", std::monostate{},
                  "allowed values: HW(m), HW(m_xor_k), HW(y), y(0)..y(7)",
                  PropertyEffect{
                      .kind = PropertyEffectKind::PayloadOutput,
                      .scope = PropertyInvalidationScope::Downstream,
                      .output_pads = {"hypotheses"},
                  }),
              PropertySpec(
                  "guess_values", IntList{},
                  "AES byte guess domain; [] means all values 0..255", "",
                  std::monostate{}, "list of integers in [0,255]",
                  PropertyEffect{
                      .kind = PropertyEffectKind::PayloadOutput,
                      .scope = PropertyInvalidationScope::Downstream,
                      .output_pads = {"hypotheses"},
                  }),
          },
      .keywords = {"aes", "sca", "cpa", "hypothesis", "leakage",
                   aes_leakage_model_id, "crypto"},
      .metadata_set_by_element =
          {
              make_element_metadata_descriptor(
                  "routing.element", std::string(),
                  "element instance name that produced the hypothesis buffer",
                  {"aesleakagehypothesis0"}),
              make_element_metadata_descriptor(
                  "payload.leakage.model", std::string(),
                  "AES leakage model identifier", {aes_leakage_model_id}),
              make_element_metadata_descriptor(
                  "payload.leakage.hypothesis", std::string(),
                  "hypothesis implementation identifier",
                  {aes_leakage_hypothesis_id}),
              make_element_metadata_descriptor(
                  "payload.leakage.byte_indexes", IntList{},
                  "AES byte indexes modeled by this element", {"[0]", "[3,5]"}),
              make_element_metadata_descriptor(
                  "payload.leakage.channels", StringList{},
                  "ordered leakage channels emitted for each byte/guess",
                  {
                      aes_leakage_channel_hw_m,
                      aes_leakage_channel_hw_m_xor_k,
                      aes_leakage_channel_hw_y,
                      std::string(aes_leakage_channel_y_bits[0]),
                  }),
              make_element_metadata_descriptor(
                  "payload.crypto.algorithm", std::string(),
                  "cryptographic algorithm name", {"AES"}),
              make_element_metadata_descriptor("payload.crypto.state_bytes",
                                               std::int64_t{},
                                               "AES state byte count", {"16"}),
              make_element_metadata_descriptor(
                  "payload.trace.count", std::int64_t{},
                  "number of traces represented by the hypotheses", {"50"}),
              make_element_metadata_descriptor(
                  "attack.hypothesis.algorithm", std::string(),
                  "attack hypothesis algorithm identifier", {"aes"}),
              make_element_metadata_descriptor(
                  "attack.hypothesis.round", std::string(),
                  "algorithm round modeled by the hypotheses", {"first"}),
              make_element_metadata_descriptor(
                  "attack.unit.kind", std::string(),
                  "kind of independent attack unit", {"byte"}),
              make_element_metadata_descriptor(
                  "attack.unit.indexes", IntList{},
                  "attack unit indexes in output order", {"[0]", "[3,5]"}),
              make_element_metadata_descriptor(
                  "attack.guess.kind", std::string(),
                  "kind of guess domain", {"byte"}),
              make_element_metadata_descriptor(
                  "attack.guess.count", std::int64_t{},
                  "number of candidate guesses on axis 1", {"256"}),
              make_element_metadata_descriptor(
                  "attack.guess.order", std::string(),
                  "meaning of guess axis order", {"domain"}),
              make_element_metadata_descriptor(
                  "attack.guess.values", std::string(),
                  "guess values represented by axis 1", {"0..255", "[0,255]"}),
              make_element_metadata_descriptor(
                  "attack.channel.depends_on_guess", StringList{},
                  "ordered booleans indicating whether each channel depends on "
                  "the key guess",
                  {"true", "false,true,true"}),
              make_element_metadata_descriptor(
                  "tensor.axes", std::string(),
                  "semantic tensor axes",
                  {"attack_unit,guess,trace,leakage_channel"}),
          },
  };
}

AesLeakageHypothesis::AesLeakageHypothesis(std::string name)
    : Element(std::move(name)) {
  configure_from_descriptor(descriptor());
}

std::optional<Buffer>
AesLeakageHypothesis::process(std::optional<Buffer> input) {
  if (!input) {
    throw std::invalid_argument("AesLeakageHypothesis requires plaintexts input");
  }

  ElementInputs inputs;
  inputs.emplace("plaintexts", *input);
  return process_plaintexts(*this, *input, std::move(inputs));
}

std::optional<Buffer>
AesLeakageHypothesis::process_inputs(ElementInputs inputs) {
  const auto &plaintext_buffer =
      required_input(inputs, "plaintexts", "AesLeakageHypothesis");
  return process_plaintexts(*this, plaintext_buffer, std::move(inputs));
}

} // namespace leakflow::plugins::crypto
