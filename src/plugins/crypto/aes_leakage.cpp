#include "leakflow/plugins/crypto/aes_leakage.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/crypto/aes.hpp"

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
          "AesLeakage byte_indexes values must be in [0,15]");
    }
    if (!seen.insert(index).second) {
      throw std::invalid_argument(
          "AesLeakage byte_indexes must not contain duplicates");
    }
    indexes.push_back(static_cast<std::size_t>(index));
  }

  return indexes;
}

[[nodiscard]] std::string
byte_indexes_metadata(const std::vector<std::size_t> &byte_indexes) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < byte_indexes.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << byte_indexes[index];
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

[[nodiscard]] bool
requires_keys(const std::vector<AesLeakageChannel> &channels) {
  return leakflow::crypto::aes::first_round_leakage_channels_depend_on_key(
      std::span<const AesLeakageChannel>(channels.data(), channels.size()));
}

[[nodiscard]] std::string
channels_metadata(const std::vector<AesLeakageChannel> &channels) {
  return leakflow::crypto::aes::first_round_leakage_channels_metadata(
      std::span<const AesLeakageChannel>(channels.data(), channels.size()));
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

void require_trace_alignment(const torch::Tensor &traces,
                             std::int64_t trace_count) {
  if (!traces.defined()) {
    throw std::invalid_argument("traces tensor must be defined");
  }
  if (traces.layout() != torch::kStrided) {
    throw std::invalid_argument("traces tensor must use strided layout");
  }
  if (traces.dim() != 2) {
    throw std::invalid_argument("traces tensor must have shape [N,M]");
  }
  if (traces.size(0) != trace_count) {
    throw std::invalid_argument(
        "traces tensor first dimension must match plaintext trace count");
  }
}

[[nodiscard]] torch::Tensor normalize_key_blocks(torch::Tensor keys,
                                                 std::int64_t trace_count) {
  require_uint8_strided(keys, "keys");

  if (keys.dim() == 1 && keys.size(0) == aes_state_bytes) {
    return keys.reshape({1, aes_state_bytes})
        .expand({trace_count, aes_state_bytes});
  }
  if (keys.dim() == 2 && keys.size(1) == aes_state_bytes) {
    if (keys.size(0) == 1) {
      return keys.expand({trace_count, aes_state_bytes});
    }
    if (keys.size(0) == trace_count) {
      return keys;
    }
  }

  throw std::invalid_argument(
      "keys tensor must have shape [16], [1,16], or [N,16]");
}

[[nodiscard]] torch::Tensor
compute_leakage(const std::vector<AesLeakageChannel> &channels,
                const torch::Tensor &plaintexts, const torch::Tensor *keys,
                std::span<const std::size_t> byte_indexes) {
  const auto channel_span =
      std::span<const AesLeakageChannel>(channels.data(), channels.size());
  if (keys == nullptr) {
    return leakflow::crypto::aes::first_round_leakage_at(
        plaintexts, byte_indexes, channel_span);
  }

  return leakflow::crypto::aes::first_round_leakage_at(
      *keys, plaintexts, byte_indexes, channel_span);
}

} // namespace

ElementDescriptor AesLeakage::descriptor() {
  return {
      .type_name = "AesLeakage",
      .klass = "Analyze/SCA/Crypto/LeakageModel",
      .purpose =
          "compute AES Hamming-weight leakage targets for selected state bytes",
      .input_pads =
          {
              Pad("traces", PadDirection::Input,
                  torch_tensor_caps({{leakflow::base::caps_param_rank, "2"}})),
              Pad("plaintexts", PadDirection::Input,
                  torch_tensor_caps({
                      {leakflow::base::caps_param_dtype, "uint8"},
                      {leakflow::base::caps_param_rank, "2"},
                  })),
              Pad("keys", PadDirection::Input,
                  torch_tensor_caps(
                      {{leakflow::base::caps_param_dtype, "uint8"}}),
                  PadPresence::Optional),
          },
      .output_pads =
          {
              Pad("leakage", PadDirection::Output,
                  torch_tensor_caps({
                      {leakflow::base::caps_param_dtype, "uint8"},
                      {leakflow::base::caps_param_rank, "3"},
                  })),
          },
      .property_specs =
          {
              PropertySpec(
                  "byte_indexes", all_byte_indexes(),
                  "AES state byte indexes to model; [] means all bytes",
                  "", std::monostate{}, "",
                  PropertyEffect{
                      .kind = PropertyEffectKind::PayloadOutput,
                      .scope = PropertyInvalidationScope::Downstream,
                      .output_pads = {"leakage"},
                  }),
              PropertySpec(
                  "channels",
                  StringList{aes_leakage_channel_hw_y},
                  "leakage channels to output; order controls output axis 2",
                  "", std::monostate{},
                  "allowed values: HW(m), HW(m_xor_k), HW(y)",
                  PropertyEffect{
                      .kind = PropertyEffectKind::PayloadOutput,
                      .scope = PropertyInvalidationScope::Downstream,
                      .output_pads = {"leakage"},
                  }),
          },
      .keywords = {"aes", "sca", "leakage", aes_leakage_model_id, "crypto"},
      .metadata_set_by_element =
          {
              make_element_metadata_descriptor(
                  "routing.element", std::string(),
                  "element instance name that produced the leakage buffer",
                  {"leakage"}),
              make_element_metadata_descriptor(
                  "payload.leakage.model", std::string(),
                  "leakage model identifier", {aes_leakage_model_id}),
              make_element_metadata_descriptor(
                  "payload.leakage.byte_indexes", IntList{},
                  "AES byte indexes modeled by this element", {"[0]", "[3,5]"}),
              make_element_metadata_descriptor(
                  "payload.leakage.channels", StringList{},
                  "leakage channels emitted for each byte",
                  {
                      aes_leakage_channel_hw_m,
                      aes_leakage_channel_hw_m_xor_k,
                      aes_leakage_channel_hw_y,
                  }),
              make_element_metadata_descriptor(
                  "payload.crypto.algorithm", std::string(),
                  "cryptographic algorithm name", {"AES"}),
              make_element_metadata_descriptor("payload.crypto.state_bytes",
                                               std::int64_t{},
                                               "AES state byte count", {"16"}),
              make_element_metadata_descriptor(
                  "payload.trace.count", std::int64_t{},
                  "number of traces represented by the leakage output", {"50"}),
              make_element_metadata_descriptor(
                  "payload.trace.input", std::string(),
                  "whether the required traces pad was connected",
                  {"connected", "unconnected"}, {}, "connected or unconnected"),
          },
  };
}

AesLeakage::AesLeakage(std::string name) : Element(std::move(name)) {
  configure_from_descriptor(descriptor());
}

std::optional<Buffer> AesLeakage::process(std::optional<Buffer>) {
  throw std::invalid_argument(
      "AesLeakage requires named traces and plaintexts inputs");
}

std::optional<Buffer> AesLeakage::process_inputs(ElementInputs inputs) {
  const auto &traces_buffer = required_input(inputs, "traces", "AesLeakage");
  const auto &plaintext_buffer =
      required_input(inputs, "plaintexts", "AesLeakage");
  const auto channels = leakage_channels_for(*this);

  const auto traces_payload = torch_payload_for(traces_buffer, "traces");
  const auto plaintext_payload =
      torch_payload_for(plaintext_buffer, "plaintexts");

  const auto &plaintexts = plaintext_payload->tensor();
  require_plaintext_blocks(plaintexts);
  const auto trace_count = plaintexts.size(0);
  require_trace_alignment(traces_payload->tensor(), trace_count);

  std::optional<torch::Tensor> keys;
  if (requires_keys(channels)) {
    const auto &key_buffer = required_input(inputs, "keys", "AesLeakage");
    const auto key_payload = torch_payload_for(key_buffer, "keys");
    keys = normalize_key_blocks(key_payload->tensor(), trace_count);
  }

  const auto byte_indexes = selected_byte_indexes(*this);
  auto leakage = compute_leakage(
      channels, plaintexts, keys ? &*keys : nullptr,
      std::span<const std::size_t>(byte_indexes.data(), byte_indexes.size()));

  auto payload = leakflow::base::TorchTensorPayload(std::move(leakage));
  Buffer output{payload.caps()};
  forward_metadata(inputs, profile_for_klass(element_kclass()), output, name());
  output.set_metadata("routing.element", name());
  output.set_metadata("payload.leakage.model", aes_leakage_model_id);
  output.set_metadata("payload.leakage.byte_indexes",
                      byte_indexes_metadata(byte_indexes));
  output.set_metadata("payload.leakage.channels", channels_metadata(channels));
  output.set_metadata("payload.crypto.algorithm", "AES");
  output.set_metadata("payload.crypto.state_bytes",
                      std::to_string(aes_state_bytes));
  output.set_metadata("payload.trace.count", std::to_string(trace_count));
  output.set_metadata("payload.trace.input", "connected");
  output.set_payload(
      std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));

  auto record = make_log_record(log::LogLevel::Debug, "element",
                                "computed AES leakage model");
  record.fields.emplace("payload.leakage.model", aes_leakage_model_id);
  record.fields.emplace("payload.leakage.channels",
                        output.metadata("payload.leakage.channels"));
  record.fields.emplace("byte_indexes",
                        output.metadata("payload.leakage.byte_indexes"));
  record.fields.emplace("payload.trace.count",
                        output.metadata("payload.trace.count"));
  record.fields.emplace("payload.trace.input",
                        output.metadata("payload.trace.input"));
  leakflow::log::write(std::move(record));

  return output;
}

} // namespace leakflow::plugins::crypto
