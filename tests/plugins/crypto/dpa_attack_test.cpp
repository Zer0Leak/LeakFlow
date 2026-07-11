#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plugins/crypto/crypto_elements.hpp"
#include "leakflow/plugins/crypto/descriptor_catalog.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

template <typename Exception, typename Function>
bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

leakflow::Buffer torch_buffer(torch::Tensor tensor)
{
    auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    leakflow::Buffer buffer(payload.caps());
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    return buffer;
}

leakflow::Buffer hypothesis_buffer(torch::Tensor tensor)
{
    auto buffer = torch_buffer(std::move(tensor));
    buffer.set_metadata("payload.leakage.model", "aes-first-round");
    buffer.set_metadata("payload.leakage.hypothesis", "aes-first-round-leakage-hypothesis");
    buffer.set_metadata("payload.leakage.byte_indexes", "[3]");
    buffer.set_metadata("payload.leakage.channels", "y(0)");
    buffer.set_metadata("payload.crypto.algorithm", "AES");
    buffer.set_metadata("payload.crypto.state_bytes", "16");
    buffer.set_metadata("attack.hypothesis.algorithm", "aes");
    buffer.set_metadata("attack.hypothesis.round", "first");
    buffer.set_metadata("attack.unit.kind", "byte");
    buffer.set_metadata("attack.unit.indexes", "[3]");
    buffer.set_metadata("attack.guess.kind", "byte");
    buffer.set_metadata("attack.guess.count", "2");
    buffer.set_metadata("attack.guess.order", "domain");
    buffer.set_metadata("attack.guess.values", "[7,42]");
    buffer.set_metadata("attack.channel.depends_on_guess", "true");
    return buffer;
}

} // namespace

int main()
{
    namespace crypto_plugin = leakflow::plugins::crypto;

    auto features = torch::tensor(
        {
            {0.0F, 0.0F},
            {2.0F, 1.0F},
            {10.0F, 0.0F},
            {12.0F, 1.0F},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    auto hypotheses = torch::tensor(
        {{{{0}, {0}, {1}, {1}},
            {{0}, {1}, {0}, {1}}}},
        torch::TensorOptions().dtype(torch::kUInt8));

    crypto_plugin::DpaAttack attack;
    attack.set_property("score_method", std::string("max_abs"));
    leakflow::ElementInputs attack_inputs;
    attack_inputs.emplace("features", torch_buffer(features));
    attack_inputs.emplace("hypotheses", hypothesis_buffer(hypotheses));
    const auto attack_output = attack.process_inputs(std::move(attack_inputs));
    if (!expect(attack_output.has_value(), "DpaAttack did not produce output")) {
        return 1;
    }
    if (!expect(attack_output->caps().type() == crypto_plugin::attack_scores_caps_type,
            "DpaAttack output caps were wrong")) {
        return 1;
    }
    if (!expect(attack_output->metadata("attack.method") == crypto_plugin::dpa_attack_method_id,
            "DpaAttack did not stamp method metadata")) {
        return 1;
    }
    if (!expect(attack_output->metadata("attack.accumulation.mode") == "recompute",
            "DpaAttack static auto mode was not recompute")) {
        return 1;
    }
    if (!expect(attack_output->metadata("attack.difference.method") == "mean(group1)-mean(group0)",
            "DpaAttack did not stamp difference method metadata")) {
        return 1;
    }

    const auto payload = attack_output->payload_as<crypto_plugin::AttackScoresPayload>();
    if (!expect(payload != nullptr, "DpaAttack payload type was wrong")) {
        return 1;
    }
    if (!expect(payload->scores().sizes() == c10::IntArrayRef({1, 2}), "DpaAttack scores shape was wrong")) {
        return 1;
    }
    if (!expect(!payload->correlations().has_value(), "DpaAttack should not emit a correlation tensor")) {
        return 1;
    }
    if (!expect(payload->ranking()[0][0].item<std::int64_t>() == 0,
            "DpaAttack selected the wrong best guess index")) {
        return 1;
    }
    if (!expect(payload->best_guess()[0].item<std::int64_t>() == 7,
            "DpaAttack best guess value was wrong")) {
        return 1;
    }
    if (!expect(std::abs(payload->best_score()[0].item<float>() - 10.0F) < 1.0e-5F,
            "DpaAttack best score was wrong")) {
        return 1;
    }
    if (!expect(payload->best_sample()[0].item<std::int64_t>() == 0,
            "DpaAttack best sample was wrong")) {
        return 1;
    }
    if (!expect(payload->best_channel()[0].item<std::int64_t>() == 0,
            "DpaAttack best channel was wrong")) {
        return 1;
    }
    if (!expect(payload->channel_names() == std::vector<std::string>{"y(0)"},
            "DpaAttack channel names were wrong")) {
        return 1;
    }

    crypto_plugin::DpaAttack pad_attack;
    pad_attack.set_property("score_method", std::string("max_abs"));
    leakflow::ElementInputs pad_attack_inputs;
    pad_attack_inputs.emplace("features", torch_buffer(features));
    pad_attack_inputs.emplace("hypotheses", hypothesis_buffer(hypotheses));
    auto pad_outputs = pad_attack.process_pads(std::move(pad_attack_inputs));
    if (!expect(pad_outputs.size() == 2, "DpaAttack did not emit both output pads")) {
        return 1;
    }
    const auto pad_scores = pad_outputs.find("scores");
    if (!expect(pad_scores != pad_outputs.end(), "DpaAttack process_pads did not emit scores")) {
        return 1;
    }
    const auto pad_best_difference = pad_outputs.find("best_difference");
    if (!expect(pad_best_difference != pad_outputs.end(), "DpaAttack process_pads did not emit best_difference")) {
        return 1;
    }
    const auto pad_scores_payload = pad_scores->second.payload_as<crypto_plugin::AttackScoresPayload>();
    if (!expect(pad_scores_payload != nullptr, "DpaAttack process_pads scores payload was wrong")) {
        return 1;
    }
    if (!expect(torch::allclose(pad_scores_payload->scores(), payload->scores(), 1.0e-5, 1.0e-5),
            "DpaAttack process_pads scores did not match process_inputs")) {
        return 1;
    }
    if (!expect(pad_best_difference->second.caps().type() == leakflow::base::torch_tensor_caps_type,
            "DpaAttack best_difference caps type was wrong")) {
        return 1;
    }
    if (!expect(pad_best_difference->second.metadata("payload.dpa.trace") == "best_difference",
            "DpaAttack best_difference trace metadata was wrong")) {
        return 1;
    }
    if (!expect(pad_best_difference->second.metadata("tensor.axes") == "attack_unit,sample",
            "DpaAttack best_difference axes metadata was wrong")) {
        return 1;
    }
    const auto best_difference_payload =
        pad_best_difference->second.payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(best_difference_payload != nullptr, "DpaAttack best_difference payload was wrong")) {
        return 1;
    }
    if (!expect(best_difference_payload->tensor().sizes() == c10::IntArrayRef({1, 2}),
            "DpaAttack best_difference shape was wrong")) {
        return 1;
    }
    if (!expect(best_difference_payload->tensor().scalar_type() == torch::kFloat32,
            "DpaAttack best_difference dtype was wrong")) {
        return 1;
    }
    if (!expect(best_difference_payload->tensor().device().is_cpu(),
            "DpaAttack best_difference device was wrong")) {
        return 1;
    }
    const auto expected_best_difference = torch::tensor({{10.0F, 0.0F}}, torch::TensorOptions().dtype(torch::kFloat32));
    if (!expect(torch::allclose(best_difference_payload->tensor(), expected_best_difference, 1.0e-5, 1.0e-5),
            "DpaAttack best_difference values were wrong")) {
        return 1;
    }

    crypto_plugin::DpaAttack incremental_attack;
    incremental_attack.set_property("accumulation_mode", std::string("incremental"));
    incremental_attack.start();
    if (!expect(!incremental_attack.can_replay(), "DpaAttack incremental mode was marked replay-safe")) {
        return 1;
    }
    auto first_features = features.index({torch::indexing::Slice(0, 2)});
    auto second_features = features.index({torch::indexing::Slice(2, 4)});
    auto first_hypotheses = hypotheses.index({torch::indexing::Slice(), torch::indexing::Slice(),
        torch::indexing::Slice(0, 2), torch::indexing::Slice()});
    auto second_hypotheses = hypotheses.index({torch::indexing::Slice(), torch::indexing::Slice(),
        torch::indexing::Slice(2, 4), torch::indexing::Slice()});

    leakflow::ElementInputs first_inputs;
    first_inputs.emplace("features", torch_buffer(first_features));
    first_inputs.emplace("hypotheses", hypothesis_buffer(first_hypotheses));
    (void)incremental_attack.process_inputs(std::move(first_inputs));
    leakflow::ElementInputs second_inputs;
    second_inputs.emplace("features", torch_buffer(second_features));
    second_inputs.emplace("hypotheses", hypothesis_buffer(second_hypotheses));
    auto incremental_outputs = incremental_attack.process_pads(std::move(second_inputs));
    const auto incremental_output = incremental_outputs.find("scores");
    if (!expect(incremental_output != incremental_outputs.end(),
            "DpaAttack incremental process_pads did not emit scores")) {
        return 1;
    }
    const auto incremental_payload = incremental_output->second.payload_as<crypto_plugin::AttackScoresPayload>();
    if (!expect(incremental_payload != nullptr, "DpaAttack incremental payload was wrong")) {
        return 1;
    }
    if (!expect(torch::allclose(incremental_payload->scores(), payload->scores(), 1.0e-5, 1.0e-5),
            "DpaAttack incremental scores did not match recompute scores")) {
        return 1;
    }
    if (!expect(incremental_output->second.metadata("attack.accumulation.mode") == "incremental",
            "DpaAttack incremental mode metadata was wrong")) {
        return 1;
    }
    if (!expect(incremental_payload->observation_count() == 4,
            "DpaAttack incremental observation count was wrong")) {
        return 1;
    }
    const auto incremental_best_difference = incremental_outputs.find("best_difference");
    if (!expect(incremental_best_difference != incremental_outputs.end(),
            "DpaAttack incremental process_pads did not emit best_difference")) {
        return 1;
    }
    const auto incremental_best_difference_payload =
        incremental_best_difference->second.payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(incremental_best_difference_payload != nullptr,
            "DpaAttack incremental best_difference payload was wrong")) {
        return 1;
    }
    if (!expect(torch::allclose(
            incremental_best_difference_payload->tensor(), expected_best_difference, 1.0e-5, 1.0e-5),
            "DpaAttack incremental best_difference values were wrong")) {
        return 1;
    }

    crypto_plugin::AttackStats stats;
    auto key = torch::tensor({0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        torch::TensorOptions().dtype(torch::kUInt8));
    leakflow::ElementInputs stats_inputs;
    stats_inputs.emplace("scores", *attack_output);
    stats_inputs.emplace("truth", torch_buffer(key));
    const auto stats_output = stats.process_inputs(std::move(stats_inputs));
    const auto stats_payload = stats_output->payload_as<crypto_plugin::AttackStatsPayload>();
    if (!expect(stats_payload != nullptr, "AttackStats rejected DpaAttack scores")) {
        return 1;
    }
    if (!expect(stats_payload->has_truth(), "AttackStats did not compute truth diagnostics for DPA")) {
        return 1;
    }
    if (!expect((*stats_payload->success())[0].item<bool>(), "AttackStats did not mark DPA success")) {
        return 1;
    }
    if (!expect(stats_output->metadata("attack.method") == crypto_plugin::dpa_attack_method_id,
            "AttackStats did not preserve DPA method metadata")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([&features] {
            auto non_binary = torch::tensor(
                {{{{0}, {2}, {1}, {1}},
                    {{0}, {1}, {0}, {1}}}},
                torch::TensorOptions().dtype(torch::kUInt8));
            crypto_plugin::DpaAttack bad_attack;
            leakflow::ElementInputs bad_inputs;
            bad_inputs.emplace("features", torch_buffer(features));
            bad_inputs.emplace("hypotheses", hypothesis_buffer(non_binary));
            (void)bad_attack.process_inputs(std::move(bad_inputs));
        }),
            "DpaAttack accepted non-binary hypotheses")) {
        return 1;
    }

    const auto descriptors = crypto_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "crypto plugin descriptor count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 12, "crypto plugin element count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[3].type_name == "DpaAttack",
            "DpaAttack descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[3].output_pads.size() == 2,
            "DpaAttack descriptor output pad count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[3].output_pads[0].name() == "scores",
            "DpaAttack descriptor scores pad was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[3].output_pads[1].name() == "best_difference",
            "DpaAttack descriptor best_difference pad was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[3].output_pads[1].presence() == leakflow::PadPresence::Optional,
            "DpaAttack descriptor best_difference pad was not optional")) {
        return 1;
    }

    return 0;
}
