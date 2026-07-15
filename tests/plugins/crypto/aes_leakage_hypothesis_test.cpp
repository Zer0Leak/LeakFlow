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

  const std::vector<std::size_t> byte_indexes{0, 2};
  const std::vector<leakflow::crypto::aes::FirstRoundLeakageChannel>
      default_channels{leakflow::crypto::aes::FirstRoundLeakageChannel::HwY};
  const auto default_channel_span =
      std::span<const leakflow::crypto::aes::FirstRoundLeakageChannel>(
          default_channels.data(), default_channels.size());

  crypto_plugin::AesLeakageHypothesis hypothesis;
  hypothesis.set_property("byte_indexes", leakflow::Units::of({0, 2}));

  leakflow::ElementInputs inputs;
  inputs.emplace("plaintexts", torch_buffer(plaintexts));
  auto output = hypothesis.process_inputs(std::move(inputs));
  if (!expect(output.has_value(),
              "AesLeakageHypothesis did not produce an output buffer")) {
    return 1;
  }
  if (!expect(output->caps().type() == "leakflow/torch-tensor",
              "AesLeakageHypothesis emitted wrong caps type")) {
    return 1;
  }
  if (!expect(output->metadata("payload.leakage.hypothesis") ==
                  crypto_plugin::aes_leakage_hypothesis_id,
              "AesLeakageHypothesis did not stamp hypothesis metadata")) {
    return 1;
  }
  if (!expect(output->metadata("payload.leakage.model") ==
                  crypto_plugin::aes_leakage_model_id,
              "AesLeakageHypothesis did not stamp leakage model metadata")) {
    return 1;
  }
  if (!expect(output->metadata("payload.leakage.byte_indexes") == "[0,2]",
              "AesLeakageHypothesis did not stamp byte index metadata")) {
    return 1;
  }
  if (!expect(output->metadata("payload.leakage.channels") ==
                  crypto_plugin::aes_leakage_channel_hw_y,
              "AesLeakageHypothesis default channel metadata was wrong")) {
    return 1;
  }
  if (!expect(output->metadata("attack.hypothesis.algorithm") == "aes",
              "AesLeakageHypothesis did not stamp attack algorithm metadata")) {
    return 1;
  }
  if (!expect(output->metadata("attack.unit.kind") == "byte",
              "AesLeakageHypothesis did not stamp attack unit metadata")) {
    return 1;
  }
  if (!expect(output->metadata("attack.unit.count") == "2",
              "AesLeakageHypothesis unit count was wrong")) {
    return 1;
  }
  if (!expect(output->metadata("attack.guess.count") == "256",
              "AesLeakageHypothesis default guess count was wrong")) {
    return 1;
  }
  if (!expect(output->metadata("attack.guess.values") == "0..255",
              "AesLeakageHypothesis default guess values metadata was wrong")) {
    return 1;
  }
  if (!expect(output->metadata("attack.channel.depends_on_guess") == "true",
              "AesLeakageHypothesis default channel dependency was wrong")) {
    return 1;
  }
  if (!expect(output->metadata("tensor.axes") ==
                  "attack_unit,guess,trace,leakage_channel",
              "AesLeakageHypothesis tensor axis metadata was wrong")) {
    return 1;
  }
  if (!expect(output->metadata("payload.layout") == "unit/guess/trace/channel",
              "AesLeakageHypothesis payload layout was wrong")) {
    return 1;
  }

  const auto payload = output->payload_as<leakflow::base::TorchTensorPayload>();
  if (!expect(payload != nullptr,
              "AesLeakageHypothesis output payload type was wrong")) {
    return 1;
  }
  if (!expect(payload->dtype() == torch::kUInt8,
              "AesLeakageHypothesis output dtype was wrong")) {
    return 1;
  }
  if (!expect(payload->shape()[0] == 2 && payload->shape()[1] == 256 &&
                  payload->shape()[2] == 3 && payload->shape()[3] == 1,
              "AesLeakageHypothesis default output shape was wrong")) {
    return 1;
  }

  const auto full_guesses =
      torch::arange(0, 256, torch::TensorOptions().dtype(torch::kLong))
          .to(torch::kUInt8);
  const auto expected_default =
      leakflow::crypto::aes::first_round_leakage_hypotheses_at(
          full_guesses, plaintexts,
          std::span<const std::size_t>(byte_indexes.data(),
                                       byte_indexes.size()),
          default_channel_span);
  if (!expect(torch::equal(payload->tensor(), expected_default),
              "AesLeakageHypothesis default output values were wrong")) {
    return 1;
  }

  crypto_plugin::AesLeakageHypothesis combined_hypothesis;
  combined_hypothesis.set_property("byte_indexes", leakflow::Units::of({1}));
  combined_hypothesis.set_property(
      "channels", leakflow::StringList{
                      crypto_plugin::aes_leakage_channel_hw_m,
                      crypto_plugin::aes_leakage_channel_hw_m_xor_k,
                      crypto_plugin::aes_leakage_channel_hw_y,
                      std::string(crypto_plugin::aes_leakage_channel_y_bits[0]),
                  });
  combined_hypothesis.set_property("guess_values", leakflow::IntList{0, 0x7e});
  leakflow::ElementInputs combined_inputs;
  combined_inputs.emplace("plaintexts", torch_buffer(plaintexts));
  const auto combined_output =
      combined_hypothesis.process_inputs(std::move(combined_inputs));
  if (!expect(combined_output.has_value(),
              "AesLeakageHypothesis rejected a multi-channel hypothesis")) {
    return 1;
  }
  if (!expect(combined_output->metadata("payload.leakage.channels") ==
                  "HW(m),HW(m_xor_k),HW(y),y(0)",
              "AesLeakageHypothesis multi-channel metadata was wrong")) {
    return 1;
  }
  if (!expect(combined_output->metadata("attack.channel.depends_on_guess") ==
                  "false,true,true,true",
              "AesLeakageHypothesis channel dependency metadata was wrong")) {
    return 1;
  }
  if (!expect(combined_output->metadata("attack.guess.count") == "2",
              "AesLeakageHypothesis custom guess count was wrong")) {
    return 1;
  }
  if (!expect(combined_output->metadata("attack.guess.values") == "[0,126]",
              "AesLeakageHypothesis custom guess values metadata was wrong")) {
    return 1;
  }

  const auto combined_payload =
      combined_output->payload_as<leakflow::base::TorchTensorPayload>();
  if (!expect(combined_payload != nullptr &&
                  combined_payload->shape()[0] == 1 &&
                  combined_payload->shape()[1] == 2 &&
                  combined_payload->shape()[2] == 3 &&
                  combined_payload->shape()[3] == 4,
              "AesLeakageHypothesis multi-channel output shape was wrong")) {
    return 1;
  }

  const std::vector<std::size_t> byte_one{1};
  const auto custom_guesses = torch::tensor(
      {0x00, 0x7e}, torch::TensorOptions().dtype(torch::kUInt8));
  const std::vector<leakflow::crypto::aes::FirstRoundLeakageChannel>
      combined_channels{
          leakflow::crypto::aes::FirstRoundLeakageChannel::HwM,
          leakflow::crypto::aes::FirstRoundLeakageChannel::HwMXorK,
          leakflow::crypto::aes::FirstRoundLeakageChannel::HwY,
          leakflow::crypto::aes::FirstRoundLeakageChannel::YBit0,
      };
  const auto expected_combined =
      leakflow::crypto::aes::first_round_leakage_hypotheses_at(
          custom_guesses, plaintexts,
          std::span<const std::size_t>(byte_one.data(), byte_one.size()),
          std::span<const leakflow::crypto::aes::FirstRoundLeakageChannel>(
              combined_channels.data(), combined_channels.size()));
  if (!expect(torch::equal(combined_payload->tensor(), expected_combined),
              "AesLeakageHypothesis multi-channel output values were wrong")) {
    return 1;
  }

  crypto_plugin::AesLeakageHypothesis direct_hypothesis;
  direct_hypothesis.set_property("byte_indexes", leakflow::Units::of({0}));
  direct_hypothesis.set_property("guess_values", leakflow::IntList{0});
  const auto direct_output = direct_hypothesis.process(torch_buffer(plaintexts));
  if (!expect(direct_output.has_value(),
              "AesLeakageHypothesis direct process produced no output")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([] {
                crypto_plugin::AesLeakageHypothesis missing_plaintexts;
                leakflow::ElementInputs missing_inputs;
                (void)missing_plaintexts.process_inputs(
                    std::move(missing_inputs));
              }),
              "AesLeakageHypothesis accepted missing plaintexts input")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([&plaintexts] {
                crypto_plugin::AesLeakageHypothesis invalid_guess;
                invalid_guess.set_property("guess_values",
                                           leakflow::IntList{0, 256});
                leakflow::ElementInputs invalid_inputs;
                invalid_inputs.emplace("plaintexts", torch_buffer(plaintexts));
                (void)invalid_guess.process_inputs(std::move(invalid_inputs));
              }),
              "AesLeakageHypothesis accepted an out-of-range guess")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([&plaintexts] {
                crypto_plugin::AesLeakageHypothesis duplicate_guess;
                duplicate_guess.set_property("guess_values",
                                             leakflow::IntList{7, 7});
                leakflow::ElementInputs duplicate_inputs;
                duplicate_inputs.emplace("plaintexts",
                                         torch_buffer(plaintexts));
                (void)duplicate_guess.process_inputs(
                    std::move(duplicate_inputs));
              }),
              "AesLeakageHypothesis accepted duplicate guesses")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([&plaintexts] {
                crypto_plugin::AesLeakageHypothesis invalid_channels;
                invalid_channels.set_property("channels",
                                              leakflow::StringList{"HW(z)"});
                leakflow::ElementInputs invalid_inputs;
                invalid_inputs.emplace("plaintexts", torch_buffer(plaintexts));
                (void)invalid_channels.process_inputs(
                    std::move(invalid_inputs));
              }),
              "AesLeakageHypothesis accepted invalid channels")) {
    return 1;
  }

  if (!expect(throws_exception<std::invalid_argument>([] {
                crypto_plugin::AesLeakageHypothesis bad_plaintexts;
                leakflow::ElementInputs bad_inputs;
                bad_inputs.emplace(
                    "plaintexts",
                    torch_buffer(torch::zeros(
                        {3, 15}, torch::TensorOptions().dtype(torch::kUInt8))));
                (void)bad_plaintexts.process_inputs(std::move(bad_inputs));
              }),
              "AesLeakageHypothesis accepted bad plaintext shape")) {
    return 1;
  }

  const auto descriptors = crypto_plugin::plugin_descriptors();
  if (!expect(descriptors.size() == 1,
              "crypto plugin descriptor count changed")) {
    return 1;
  }
  if (!expect(descriptors[0].elements.size() == 12,
              "crypto element descriptor count was wrong")) {
    return 1;
  }
  if (!expect(descriptors[0].elements[1].type_name == "AesLeakageHypothesis",
              "AesLeakageHypothesis descriptor type name was wrong")) {
    return 1;
  }
  if (!expect(descriptors[0].elements[1].input_pads[0].name() == "plaintexts",
              "AesLeakageHypothesis plaintexts pad was wrong")) {
    return 1;
  }
  if (!expect(descriptors[0].elements[1].output_pads[0].name() == "hypotheses",
              "AesLeakageHypothesis output pad was wrong")) {
    return 1;
  }

  bool saw_guess_values_effect = false;
  for (const auto &property : descriptors[0].elements[1].property_specs) {
    if (property.name == "guess_values") {
      saw_guess_values_effect =
          property.effect.kind == leakflow::PropertyEffectKind::PayloadOutput &&
          property.effect.scope ==
              leakflow::PropertyInvalidationScope::Downstream &&
          property.effect.output_pads ==
              std::vector<std::string>{"hypotheses"};
    }
  }
  if (!expect(saw_guess_values_effect,
              "AesLeakageHypothesis guess_values effect was not declared")) {
    return 1;
  }

  return 0;
}
