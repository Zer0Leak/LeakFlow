#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plugins/crypto/crypto_elements.hpp"
#include "leakflow/plugins/crypto/descriptor_catalog.hpp"

#include <algorithm>
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
    buffer.set_metadata("payload.leakage.channels", "HW(y)");
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
            {1.0F, 2.0F},
            {2.0F, 1.0F},
            {3.0F, 3.0F},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    auto hypotheses = torch::tensor(
        {{{{0}, {1}, {2}, {3}},
            {{0}, {1}, {0}, {1}}}},
        torch::TensorOptions().dtype(torch::kUInt8));

    crypto_plugin::CpaAttack attack;
    attack.set_property("score_method", std::string("max_abs"));
    attack.set_property("emit_correlations", true);
    leakflow::ElementInputs attack_inputs;
    attack_inputs.emplace("features", torch_buffer(features));
    attack_inputs.emplace("hypotheses", hypothesis_buffer(hypotheses));
    const auto attack_output = attack.process_inputs(std::move(attack_inputs));
    if (!expect(attack_output.has_value(), "CpaAttack did not produce output")) {
        return 1;
    }
    if (!expect(attack_output->caps().type() == crypto_plugin::attack_scores_caps_type,
            "CpaAttack output caps were wrong")) {
        return 1;
    }
    if (!expect(attack_output->metadata("attack.method") == crypto_plugin::cpa_attack_method_id,
            "CpaAttack did not stamp method metadata")) {
        return 1;
    }
    if (!expect(attack_output->metadata("attack.correlation.mode") == "recompute",
            "CpaAttack static auto mode was not recompute")) {
        return 1;
    }
    if (!expect(attack_output->metadata("attack.correlations.emitted") == "true",
            "CpaAttack did not report emitted correlations")) {
        return 1;
    }

    const auto payload = attack_output->payload_as<crypto_plugin::AttackScoresPayload>();
    if (!expect(payload != nullptr, "CpaAttack payload type was wrong")) {
        return 1;
    }
    if (!expect(payload->scores().sizes() == c10::IntArrayRef({1, 2}), "CpaAttack scores shape was wrong")) {
        return 1;
    }
    if (!expect(payload->correlations().has_value(), "CpaAttack did not retain correlations")) {
        return 1;
    }
    if (!expect(payload->correlations()->sizes() == c10::IntArrayRef({1, 2, 1, 2}),
            "CpaAttack correlations shape was wrong")) {
        return 1;
    }
    if (!expect(payload->ranking()[0][0].item<std::int64_t>() == 0,
            "CpaAttack selected the wrong best guess index")) {
        return 1;
    }
    if (!expect(payload->best_guess()[0].item<std::int64_t>() == 7,
            "CpaAttack best guess value was wrong")) {
        return 1;
    }
    if (!expect(payload->best_sample()[0].item<std::int64_t>() == 0,
            "CpaAttack best sample was wrong")) {
        return 1;
    }
    if (!expect(payload->best_channel()[0].item<std::int64_t>() == 0,
            "CpaAttack best channel was wrong")) {
        return 1;
    }

    crypto_plugin::CpaAttack incremental_attack;
    incremental_attack.set_property("correlation_mode", std::string("incremental"));
    incremental_attack.start();
    if (!expect(!incremental_attack.can_replay(), "CpaAttack incremental mode was marked replay-safe")) {
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
    const auto incremental_output = incremental_attack.process_inputs(std::move(second_inputs));
    const auto incremental_payload = incremental_output->payload_as<crypto_plugin::AttackScoresPayload>();
    if (!expect(incremental_payload != nullptr, "CpaAttack incremental payload was wrong")) {
        return 1;
    }
    if (!expect(torch::allclose(incremental_payload->scores(), payload->scores(), 1.0e-5, 1.0e-5),
            "CpaAttack incremental scores did not match recompute scores")) {
        return 1;
    }
    if (!expect(incremental_output->metadata("attack.correlation.mode") == "incremental",
            "CpaAttack incremental mode metadata was wrong")) {
        return 1;
    }
    if (!expect(incremental_payload->observation_count() == 4,
            "CpaAttack incremental observation count was wrong")) {
        return 1;
    }

    crypto_plugin::AttackStats stats;
    auto key = torch::tensor({0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        torch::TensorOptions().dtype(torch::kUInt8));
    leakflow::ElementInputs stats_inputs;
    stats_inputs.emplace("scores", *attack_output);
    stats_inputs.emplace("truth", torch_buffer(key));
    const auto stats_output = stats.process_inputs(std::move(stats_inputs));
    if (!expect(stats_output.has_value(), "AttackStats did not produce output")) {
        return 1;
    }
    if (!expect(stats_output->caps().type() == crypto_plugin::attack_stats_caps_type,
            "AttackStats output caps were wrong")) {
        return 1;
    }
    const auto stats_payload = stats_output->payload_as<crypto_plugin::AttackStatsPayload>();
    if (!expect(stats_payload != nullptr, "AttackStats payload type was wrong")) {
        return 1;
    }
    if (!expect(stats_payload->has_truth(), "AttackStats with truth was missing truth diagnostics")) {
        return 1;
    }
    if (!expect((*stats_payload->true_rank())[0].item<std::int64_t>() == 1,
            "AttackStats true rank was wrong")) {
        return 1;
    }
    if (!expect((*stats_payload->true_guess())[0].item<std::int64_t>() == 7,
            "AttackStats true guess was wrong")) {
        return 1;
    }
    if (!expect((*stats_payload->success())[0].item<bool>(),
            "AttackStats success flag was wrong")) {
        return 1;
    }
    if (!expect(stats_payload->best_sample()[0].item<std::int64_t>() == 0,
            "AttackStats best sample was not copied from attack result")) {
        return 1;
    }
    if (!expect(stats_payload->best_channel()[0].item<std::int64_t>() == 0,
            "AttackStats best channel was not copied from attack result")) {
        return 1;
    }
    if (!expect(stats_payload->top_k() == 2,
            "AttackStats top_k was not clipped to the available guess count")) {
        return 1;
    }
    if (!expect(stats_payload->confidence_metrics()
                    == std::vector<std::string>{"relative_margin", "z_score", "robust_z_score"},
            "AttackStats default confidence metrics were wrong")) {
        return 1;
    }
    if (!expect(stats_payload->topk_guess()[0][0].item<std::int64_t>() == 7,
            "AttackStats topk guess was wrong")) {
        return 1;
    }
    if (!expect(stats_payload->topk_guess().sizes() == c10::IntArrayRef({1, 2}),
            "AttackStats topk_guess shape was wrong")) {
        return 1;
    }
    const auto top1_score = stats_payload->topk_score()[0][0].item<double>();
    const auto top2_score = stats_payload->topk_score()[0][1].item<double>();
    const auto expected_relative_margin = (top1_score - top2_score) / std::max(std::abs(top1_score), 1.0e-12);
    if (!expect(std::abs(stats_payload->score_gap()[0].item<double>() - expected_relative_margin) < 1.0e-9,
            "AttackStats score_gap was not the top-1 relative margin")) {
        return 1;
    }
    if (!expect(std::abs(stats_payload->topk_relative_margin()[0][0].item<double>() - expected_relative_margin)
                < 1.0e-9,
            "AttackStats topk relative margin was wrong")) {
        return 1;
    }
    if (!expect(stats_output->metadata("attack.stats.score_gap") == "relative_margin",
            "AttackStats score_gap metadata was wrong")) {
        return 1;
    }
    if (!expect(stats_output->metadata("attack.method") == crypto_plugin::cpa_attack_method_id,
            "AttackStats did not preserve attack method metadata")) {
        return 1;
    }
    if (!expect(stats_output->metadata("attack.score.method") == "max_abs",
            "AttackStats did not preserve attack score metadata")) {
        return 1;
    }

    crypto_plugin::AttackStats custom_stats;
    custom_stats.set_property("top_k", std::int64_t{1});
    custom_stats.set_property("confidence_metrics", leakflow::StringList{"margin", "top-k-separation"});
    leakflow::ElementInputs custom_stats_inputs;
    custom_stats_inputs.emplace("scores", *attack_output);
    custom_stats_inputs.emplace("truth", torch_buffer(key));
    const auto custom_stats_output = custom_stats.process_inputs(std::move(custom_stats_inputs));
    const auto custom_stats_payload = custom_stats_output->payload_as<crypto_plugin::AttackStatsPayload>();
    if (!expect(custom_stats_payload->top_k() == 1, "AttackStats custom top_k was wrong")) {
        return 1;
    }
    if (!expect(custom_stats_payload->confidence_metrics() == std::vector<std::string>{"margin", "top_k_separation"},
            "AttackStats custom confidence metrics were wrong")) {
        return 1;
    }

    crypto_plugin::AttackStatsToPlotAnnotations annotations;
    annotations.set_property("precision", std::int64_t{3});
    const auto annotation_output = annotations.process(*stats_output);
    if (!expect(annotation_output.has_value(), "AttackStatsToPlotAnnotations did not produce output")) {
        return 1;
    }
    const auto annotation_payload = annotation_output->payload_as<leakflow::base::PlotAnnotationPayload>();
    if (!expect(annotation_payload != nullptr, "AttackStatsToPlotAnnotations payload type was wrong")) {
        return 1;
    }
    if (!expect(annotation_payload->annotation_count() == 1,
            "AttackStatsToPlotAnnotations annotation count was wrong")) {
        return 1;
    }
    if (!expect(annotation_payload->annotation(0).sample_index == 0,
            "AttackStatsToPlotAnnotations sample index was wrong")) {
        return 1;
    }
    if (!expect(annotation_payload->annotation(0).fields[1].second == "true",
            "AttackStatsToPlotAnnotations success field was wrong")) {
        return 1;
    }
    if (!expect(annotation_payload->annotation(0).label == "unit 3 [HW(y)]",
            "AttackStatsToPlotAnnotations success label was wrong")) {
        return 1;
    }
    if (!expect(!annotation_payload->annotation(0).text.starts_with("PASS."),
            "AttackStatsToPlotAnnotations success text kept the PASS prefix")) {
        return 1;
    }
    if (!expect(annotation_payload->annotation(0).fields[2].second == "7",
            "AttackStatsToPlotAnnotations guess field was wrong")) {
        return 1;
    }
    if (!expect(annotation_payload->annotation(0).norm_value && *annotation_payload->annotation(0).norm_value > 0.0,
            "AttackStatsToPlotAnnotations success norm value was not positive")) {
        return 1;
    }
    if (!expect(annotation_payload->annotation(0).marker == "square",
            "AttackStatsToPlotAnnotations success marker should be square")) {
        return 1;
    }
    if (!expect(annotation_output->metadata("payload.annotation.success_source") == "stats",
            "AttackStatsToPlotAnnotations success source metadata was wrong")) {
        return 1;
    }
    if (!expect(annotation_output->metadata("attack.method") == crypto_plugin::cpa_attack_method_id,
            "AttackStatsToPlotAnnotations did not preserve attack method metadata")) {
        return 1;
    }
    if (!expect(annotation_output->metadata("attack.score.method") == "max_abs",
            "AttackStatsToPlotAnnotations did not preserve attack score metadata")) {
        return 1;
    }
    if (!expect(annotation_output->metadata("attack.stats.score_gap") == "relative_margin",
            "AttackStatsToPlotAnnotations did not preserve attack stats metadata")) {
        return 1;
    }

    crypto_plugin::AttackStats failed_stats;
    auto wrong_key = torch::tensor({0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        torch::TensorOptions().dtype(torch::kUInt8));
    leakflow::ElementInputs failed_stats_inputs;
    failed_stats_inputs.emplace("scores", *attack_output);
    failed_stats_inputs.emplace("truth", torch_buffer(wrong_key));
    const auto failed_stats_output = failed_stats.process_inputs(std::move(failed_stats_inputs));
    const auto failed_stats_payload = failed_stats_output->payload_as<crypto_plugin::AttackStatsPayload>();
    if (!expect(!(*failed_stats_payload->success())[0].item<bool>(),
            "AttackStats failed case unexpectedly succeeded")) {
        return 1;
    }

    const auto rich_annotation_output = annotations.process(*failed_stats_output);
    const auto rich_annotation_payload = rich_annotation_output->payload_as<leakflow::base::PlotAnnotationPayload>();
    // Failure is now encoded by the marker shape (x), not a negative value: the
    // height stays positive (score magnitude) so failed units are not pushed down.
    if (!expect(rich_annotation_payload->annotation(0).norm_value
                && *rich_annotation_payload->annotation(0).norm_value > 0.0,
            "AttackStatsToPlotAnnotations failed stats norm value should stay positive")) {
        return 1;
    }
    if (!expect(rich_annotation_payload->annotation(0).marker == "x",
            "AttackStatsToPlotAnnotations failed stats marker should be x")) {
        return 1;
    }
    if (!expect(rich_annotation_payload->annotation(0).fields[1].second == "false",
            "AttackStatsToPlotAnnotations failed stats success field was wrong")) {
        return 1;
    }
    if (!expect(rich_annotation_payload->annotation(0).fields[5].second == "2",
            "AttackStatsToPlotAnnotations failed stats true rank field was wrong")) {
        return 1;
    }
    if (!expect(rich_annotation_payload->annotation(0).fields[6].second == "42",
            "AttackStatsToPlotAnnotations failed stats true guess field was wrong")) {
        return 1;
    }
    if (!expect(rich_annotation_payload->annotation(0).label == "unit 3 [HW(y)]",
            "AttackStatsToPlotAnnotations failed label was wrong")) {
        return 1;
    }
    if (!expect(!rich_annotation_payload->annotation(0).text.starts_with("FAIL."),
            "AttackStatsToPlotAnnotations failed text kept the FAIL prefix")) {
        return 1;
    }
    if (!expect(rich_annotation_output->metadata("payload.annotation.success_source") == "stats",
            "AttackStatsToPlotAnnotations success source metadata was wrong")) {
        return 1;
    }
    if (!expect(rich_annotation_payload->annotation(0).fields.back().first == "correct key",
            "AttackStatsToPlotAnnotations failed stats missed the correct-key field")) {
        return 1;
    }

    // AttackStats without a truth input: GE/PGE-style fields are skipped, but
    // every truth-independent statistic is still produced.
    crypto_plugin::AttackStats no_truth_stats;
    leakflow::ElementInputs no_truth_inputs;
    no_truth_inputs.emplace("scores", *attack_output);
    const auto no_truth_output = no_truth_stats.process_inputs(std::move(no_truth_inputs));
    if (!expect(no_truth_output.has_value(), "AttackStats without truth did not produce output")) {
        return 1;
    }
    if (!expect(no_truth_output->caps().type() == crypto_plugin::attack_stats_caps_type,
            "AttackStats without truth output caps were wrong")) {
        return 1;
    }
    const auto no_truth_payload = no_truth_output->payload_as<crypto_plugin::AttackStatsPayload>();
    if (!expect(no_truth_payload != nullptr, "AttackStats without truth payload type was wrong")) {
        return 1;
    }
    if (!expect(!no_truth_payload->has_truth(), "AttackStats without truth still reported truth diagnostics")) {
        return 1;
    }
    if (!expect(!no_truth_payload->true_rank().has_value() && !no_truth_payload->true_guess().has_value()
                    && !no_truth_payload->true_score().has_value() && !no_truth_payload->success().has_value(),
            "AttackStats without truth exposed truth tensors")) {
        return 1;
    }
    if (!expect(no_truth_payload->unit_count() == 1, "AttackStats without truth unit count was wrong")) {
        return 1;
    }
    if (!expect(no_truth_payload->top_k() == 2, "AttackStats without truth top_k was wrong")) {
        return 1;
    }
    if (!expect(no_truth_payload->topk_guess()[0][0].item<std::int64_t>() == 7,
            "AttackStats without truth topk guess was wrong")) {
        return 1;
    }
    if (!expect(std::abs(no_truth_payload->score_gap()[0].item<double>() - expected_relative_margin) < 1.0e-9,
            "AttackStats without truth score_gap was not the top-1 relative margin")) {
        return 1;
    }
    if (!expect(no_truth_payload->best_sample()[0].item<std::int64_t>() == 0,
            "AttackStats without truth best sample was wrong")) {
        return 1;
    }
    if (!expect(no_truth_output->metadata("attack.stats.has_truth") == "false",
            "AttackStats without truth has_truth metadata was wrong")) {
        return 1;
    }
    if (!expect(!no_truth_output->has_metadata("attack.stats.rank_base")
                    && !no_truth_output->has_metadata("attack.stats.success_count"),
            "AttackStats without truth still stamped truth-only metadata")) {
        return 1;
    }
    if (!expect(no_truth_output->metadata("attack.stats.score_gap") == "relative_margin",
            "AttackStats without truth score_gap metadata was wrong")) {
        return 1;
    }

    crypto_plugin::AttackStatsToPlotAnnotations no_truth_annotations;
    const auto no_truth_annotation_output = no_truth_annotations.process(*no_truth_output);
    if (!expect(no_truth_annotation_output.has_value(),
            "AttackStatsToPlotAnnotations without truth did not produce output")) {
        return 1;
    }
    const auto no_truth_annotation_payload =
        no_truth_annotation_output->payload_as<leakflow::base::PlotAnnotationPayload>();
    if (!expect(no_truth_annotation_payload != nullptr,
            "AttackStatsToPlotAnnotations without truth payload type was wrong")) {
        return 1;
    }
    if (!expect(no_truth_annotation_payload->annotation_count() == 1,
            "AttackStatsToPlotAnnotations without truth annotation count was wrong")) {
        return 1;
    }
    if (!expect(no_truth_annotation_payload->annotation(0).norm_value
                    && *no_truth_annotation_payload->annotation(0).norm_value > 0.0,
            "AttackStatsToPlotAnnotations without truth norm value was not positive")) {
        return 1;
    }
    if (!expect(no_truth_annotation_payload->annotation(0).marker == "circle",
            "AttackStatsToPlotAnnotations without truth marker should be circle")) {
        return 1;
    }
    if (!expect(no_truth_annotation_payload->annotation(0).fields[0].first == "attack unit"
                    && no_truth_annotation_payload->annotation(0).fields[1].first == "guess",
            "AttackStatsToPlotAnnotations without truth still emitted a success field")) {
        return 1;
    }
    if (!expect(no_truth_annotation_output->metadata("payload.annotation.success_source") == "none",
            "AttackStatsToPlotAnnotations without truth success source metadata was wrong")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([&features, &hypotheses] {
            auto non_dependent = hypothesis_buffer(hypotheses);
            non_dependent.set_metadata("attack.channel.depends_on_guess", "false");
            crypto_plugin::CpaAttack bad_attack;
            leakflow::ElementInputs bad_inputs;
            bad_inputs.emplace("features", torch_buffer(features));
            bad_inputs.emplace("hypotheses", std::move(non_dependent));
            (void)bad_attack.process_inputs(std::move(bad_inputs));
        }),
            "CpaAttack accepted score_channels=guess_dependent with no dependent channels")) {
        return 1;
    }

    const auto descriptors = crypto_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "crypto plugin descriptor count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 9, "crypto plugin element count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[2].type_name == "CpaAttack",
            "CpaAttack descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[3].type_name == "DpaAttack",
            "DpaAttack descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[4].type_name == "AttackStats",
            "AttackStats descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[5].type_name == "AttackStatsToPlotAnnotations",
            "AttackStatsToPlotAnnotations descriptor type name was wrong")) {
        return 1;
    }

    return 0;
}
