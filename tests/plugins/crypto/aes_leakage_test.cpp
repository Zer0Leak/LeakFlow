#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/crypto/aes.hpp"
#include "leakflow/plugins/crypto/crypto_elements.hpp"
#include "leakflow/plugins/crypto/descriptor_catalog.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }

  return true;
}

template <typename Exception, typename Function>
bool throws_exception(Function function) {
  try {
    function();
  } catch (const Exception &) {
    return true;
  }

  return false;
}

leakflow::Buffer torch_buffer(torch::Tensor tensor) {
  auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
  leakflow::Buffer buffer(payload.caps());
  buffer.set_payload(
      std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
  return buffer;
}

torch::Tensor
select_byte_columns(const torch::Tensor &blocks,
                    const std::vector<std::size_t> &byte_indexes) {
  std::vector<std::int64_t> indexes;
  indexes.reserve(byte_indexes.size());
  for (const auto index : byte_indexes) {
    indexes.push_back(static_cast<std::int64_t>(index));
  }

  const auto index_tensor = torch::tensor(
      indexes,
      torch::TensorOptions().dtype(torch::kLong).device(blocks.device()));
  return blocks.index_select(1, index_tensor).transpose(0, 1).contiguous();
}

} // namespace

int main() {
  namespace crypto_plugin = leakflow::plugins::crypto;

  auto plaintexts = torch::tensor(
      {
          {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
           0x0b, 0x0c, 0x0d, 0x0e, 0x0f},
          {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
           0x1b, 0x1c, 0x1d, 0x1e, 0x1f},
          {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a,
           0x2b, 0x2c, 0x2d, 0x2e, 0x2f},
      },
      torch::TensorOptions().dtype(torch::kUInt8));
  auto key = torch::tensor({0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c},
                           torch::TensorOptions().dtype(torch::kUInt8));
  auto traces = torch::arange(15, torch::TensorOptions().dtype(torch::kFloat32))
                    .reshape({3, 5});

  const std::vector<std::size_t> byte_indexes{0, 2};

  crypto_plugin::AesLeakage leakage;
  leakage.set_property("byte_indexes", leakflow::IntList{0, 2});

  leakflow::ElementInputs inputs;
  inputs.emplace("traces", torch_buffer(traces));
  inputs.emplace("plaintexts", torch_buffer(plaintexts));
  inputs.emplace("keys", torch_buffer(key));

  auto output = leakage.process_inputs(std::move(inputs));
  if (!expect(output.has_value(),
              "AesLeakage did not produce an output buffer")) {
    return 1;
  }
  if (!expect(output->caps().type() == "leakflow/torch-tensor",
              "AesLeakage emitted wrong caps type")) {
    return 1;
  }
  if (!expect(output->metadata("payload.leakage.model") ==
                  crypto_plugin::aes_leakage_model_id,
              "AesLeakage did not stamp leakage model metadata")) {
    return 1;
  }
  if (!expect(output->metadata("payload.leakage.byte_indexes") == "[0,2]",
              "AesLeakage did not stamp byte index metadata")) {
    return 1;
  }
  if (!expect(output->metadata("payload.leakage.channels") ==
                  crypto_plugin::aes_leakage_channel_hw_y,
              "AesLeakage did not stamp selected leakage channel metadata")) {
    return 1;
  }
  if (!expect(output->metadata("payload.trace.input") == "connected",
              "AesLeakage did not report connected traces input")) {
    return 1;
  }

  const auto payload = output->payload_as<leakflow::base::TorchTensorPayload>();
  if (!expect(payload != nullptr, "AesLeakage output payload type was wrong")) {
    return 1;
  }
  if (!expect(payload->dtype() == torch::kUInt8,
              "AesLeakage output dtype was wrong")) {
    return 1;
  }
  if (!expect(payload->shape()[0] == 2 && payload->shape()[1] == 3 &&
                  payload->shape()[2] == 1,
              "AesLeakage output shape was not [B,N,1]")) {
    return 1;
  }

  const auto expected_sbox = leakflow::crypto::aes::first_round_sbox_leakage_at(
      key.reshape({1, 16}).expand({3, 16}), plaintexts,
      std::span<const std::size_t>(byte_indexes.data(), byte_indexes.size()));
  const auto expected_hw_y =
      expected_sbox.hamming_weights.select(2, 1).unsqueeze(2).contiguous();
  if (!expect(
          torch::equal(payload->tensor(), expected_hw_y),
          "AesLeakage default HW(y) output values did not match AES helper")) {
    return 1;
  }

  crypto_plugin::AesLeakage plaintext_leakage;
  plaintext_leakage.set_property("byte_indexes", leakflow::IntList{0, 2});
  plaintext_leakage.set_property(
      "channels",
      leakflow::StringList{crypto_plugin::aes_leakage_channel_hw_m});
  leakflow::ElementInputs plaintext_inputs;
  plaintext_inputs.emplace("traces", torch_buffer(traces));
  plaintext_inputs.emplace("plaintexts", torch_buffer(plaintexts));
  const auto plaintext_output =
      plaintext_leakage.process_inputs(std::move(plaintext_inputs));
  if (!expect(plaintext_output.has_value(),
              "AesLeakage rejected HW(m) without keys")) {
    return 1;
  }
  if (!expect(plaintext_output->metadata("payload.leakage.channels") ==
                  crypto_plugin::aes_leakage_channel_hw_m,
              "AesLeakage HW(m) metadata channel was wrong")) {
    return 1;
  }
  const auto plaintext_payload =
      plaintext_output->payload_as<leakflow::base::TorchTensorPayload>();
  const auto expected_hw_m = leakflow::crypto::hamming_weight_u8(
                                 select_byte_columns(plaintexts, byte_indexes))
                                 .unsqueeze(2)
                                 .contiguous();
  if (!expect(plaintext_payload != nullptr &&
                  torch::equal(plaintext_payload->tensor(), expected_hw_m),
              "AesLeakage HW(m) output values were wrong")) {
    return 1;
  }

  crypto_plugin::AesLeakage add_round_key_leakage;
  add_round_key_leakage.set_property("byte_indexes", leakflow::IntList{1});
  add_round_key_leakage.set_property(
      "channels",
      leakflow::StringList{crypto_plugin::aes_leakage_channel_hw_m_xor_k});
  leakflow::ElementInputs xor_inputs;
  xor_inputs.emplace("traces", torch_buffer(traces));
  xor_inputs.emplace("plaintexts", torch_buffer(plaintexts));
  xor_inputs.emplace("keys", torch_buffer(key.reshape({1, 16})));
  const auto xor_output =
      add_round_key_leakage.process_inputs(std::move(xor_inputs));
  if (!expect(xor_output.has_value(),
              "AesLeakage rejected HW(m_xor_k) with keys")) {
    return 1;
  }
  if (!expect(xor_output->metadata("payload.leakage.channels") ==
                  crypto_plugin::aes_leakage_channel_hw_m_xor_k,
              "AesLeakage HW(m_xor_k) metadata channel was wrong")) {
    return 1;
  }
  const auto xor_payload =
      xor_output->payload_as<leakflow::base::TorchTensorPayload>();
  const std::vector<std::size_t> xor_byte_indexes{1};
  const auto normalized_key = key.reshape({1, 16}).expand({3, 16});
  const auto expected_hw_m_xor_k =
      leakflow::crypto::hamming_weight_u8(
          torch::bitwise_xor(
              select_byte_columns(plaintexts, xor_byte_indexes),
              select_byte_columns(normalized_key, xor_byte_indexes)))
          .unsqueeze(2)
          .contiguous();
  if (!expect(xor_payload != nullptr &&
                  torch::equal(xor_payload->tensor(), expected_hw_m_xor_k),
              "AesLeakage HW(m_xor_k) output values were wrong")) {
    return 1;
  }

  crypto_plugin::AesLeakage combined_leakage;
  combined_leakage.set_property("byte_indexes", leakflow::IntList{1});
  combined_leakage.set_property(
      "channels", leakflow::StringList{
                      crypto_plugin::aes_leakage_channel_hw_m,
                      crypto_plugin::aes_leakage_channel_hw_m_xor_k,
                      crypto_plugin::aes_leakage_channel_hw_y,
                  });
  leakflow::ElementInputs combined_inputs;
  combined_inputs.emplace("traces", torch_buffer(traces));
  combined_inputs.emplace("plaintexts", torch_buffer(plaintexts));
  combined_inputs.emplace("keys", torch_buffer(key.reshape({1, 16})));
  const auto combined_output =
      combined_leakage.process_inputs(std::move(combined_inputs));
  if (!expect(combined_output.has_value(),
              "AesLeakage rejected a multi-channel combination")) {
    return 1;
  }
  if (!expect(combined_output->metadata("payload.leakage.channels") ==
                  "HW(m),HW(m_xor_k),HW(y)",
              "AesLeakage multi-channel metadata was wrong")) {
    return 1;
  }
  const auto combined_payload =
      combined_output->payload_as<leakflow::base::TorchTensorPayload>();
  const auto expected_hw_m_byte_1 =
      leakflow::crypto::hamming_weight_u8(
          select_byte_columns(plaintexts, xor_byte_indexes))
          .contiguous();
  const auto expected_hw_y_byte_1 =
      leakflow::crypto::aes::first_round_sbox_leakage_at(
          normalized_key, plaintexts,
          std::span<const std::size_t>(xor_byte_indexes.data(),
                                       xor_byte_indexes.size()))
          .hamming_weights.select(2, 1)
          .contiguous();
  const auto expected_combined =
      torch::stack({expected_hw_m_byte_1, expected_hw_m_xor_k.squeeze(2),
                    expected_hw_y_byte_1},
                   2)
          .contiguous();
  if (!expect(combined_payload != nullptr &&
                  torch::equal(combined_payload->tensor(), expected_combined),
              "AesLeakage multi-channel output values were wrong")) {
    return 1;
  }

  if (!expect(
          throws_exception<std::invalid_argument>([&plaintexts, &traces] {
            crypto_plugin::AesLeakage missing_keys;
            leakflow::ElementInputs missing_key_inputs;
            missing_key_inputs.emplace("traces", torch_buffer(traces));
            missing_key_inputs.emplace("plaintexts", torch_buffer(plaintexts));
            (void)missing_keys.process_inputs(std::move(missing_key_inputs));
          }),
          "AesLeakage accepted default HW(y) without keys")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([&plaintexts] {
                crypto_plugin::AesLeakage missing_traces;
                missing_traces.set_property(
                    "channels", leakflow::StringList{
                                    crypto_plugin::aes_leakage_channel_hw_m});
                leakflow::ElementInputs missing_trace_inputs;
                missing_trace_inputs.emplace("plaintexts",
                                             torch_buffer(plaintexts));
                (void)missing_traces.process_inputs(
                    std::move(missing_trace_inputs));
              }),
              "AesLeakage accepted missing required traces input")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([&plaintexts, &key] {
                crypto_plugin::AesLeakage bad_traces;
                leakflow::ElementInputs bad_inputs;
                bad_inputs.emplace("traces",
                                   torch_buffer(torch::zeros(
                                       {2, 5}, torch::TensorOptions().dtype(
                                                   torch::kFloat32))));
                bad_inputs.emplace("plaintexts", torch_buffer(plaintexts));
                bad_inputs.emplace("keys", torch_buffer(key));
                (void)bad_traces.process_inputs(std::move(bad_inputs));
              }),
              "AesLeakage accepted traces with mismatched trace count")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([&plaintexts, &traces] {
                crypto_plugin::AesLeakage invalid_channels;
                invalid_channels.set_property("channels",
                                              leakflow::StringList{"HW(z)"});
                leakflow::ElementInputs invalid_inputs;
                invalid_inputs.emplace("traces", torch_buffer(traces));
                invalid_inputs.emplace("plaintexts", torch_buffer(plaintexts));
                (void)invalid_channels.process_inputs(
                    std::move(invalid_inputs));
              }),
              "AesLeakage accepted an invalid channels property value")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([&plaintexts, &traces] {
                crypto_plugin::AesLeakage duplicate_channels;
                duplicate_channels.set_property(
                    "channels", leakflow::StringList{
                                    crypto_plugin::aes_leakage_channel_hw_m,
                                    crypto_plugin::aes_leakage_channel_hw_m,
                                });
                leakflow::ElementInputs duplicate_inputs;
                duplicate_inputs.emplace("traces", torch_buffer(traces));
                duplicate_inputs.emplace("plaintexts",
                                         torch_buffer(plaintexts));
                (void)duplicate_channels.process_inputs(
                    std::move(duplicate_inputs));
              }),
              "AesLeakage accepted duplicate channels")) {
    return 1;
  }

  const auto descriptors = crypto_plugin::plugin_descriptors();
  if (!expect(descriptors.size() == 1,
              "crypto plugin descriptor count changed")) {
    return 1;
  }
  if (!expect(descriptors[0].elements.size() == 8,
              "crypto element descriptor count was wrong")) {
    return 1;
  }
  if (!expect(descriptors[0].elements[0].type_name == "AesLeakage",
              "AesLeakage descriptor type name was wrong")) {
    return 1;
  }
  if (!expect(descriptors[0].elements[0].input_pads[0].presence() ==
                  leakflow::PadPresence::Required,
              "AesLeakage traces pad was not required in descriptor")) {
    return 1;
  }
  if (!expect(descriptors[0].elements[0].input_pads[1].presence() ==
                  leakflow::PadPresence::Required,
              "AesLeakage plaintexts pad was not required in descriptor")) {
    return 1;
  }
  if (!expect(descriptors[0].elements[0].input_pads[2].presence() ==
                  leakflow::PadPresence::Optional,
              "AesLeakage keys pad was not optional in descriptor")) {
    return 1;
  }
  bool saw_channels_effect = false;
  for (const auto &property : descriptors[0].elements[0].property_specs) {
    if (property.name == "channels") {
      saw_channels_effect =
          property.effect.kind == leakflow::PropertyEffectKind::PayloadOutput &&
          property.effect.scope == leakflow::PropertyInvalidationScope::Downstream &&
          property.effect.output_pads == std::vector<std::string>{"leakage"};
    }
  }
  if (!expect(saw_channels_effect,
              "AesLeakage channels property effect was not declared")) {
    return 1;
  }

  return 0;
}
