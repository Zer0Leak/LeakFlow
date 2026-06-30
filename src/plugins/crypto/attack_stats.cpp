#include "leakflow/plugins/crypto/attack_stats.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/attack_payload.hpp"

#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::crypto {
namespace {

inline constexpr auto default_stats_top_k = std::int64_t{5};
inline constexpr auto confidence_epsilon = 1.0e-12;
inline constexpr auto robust_z_scale = 0.6744897501960817;

[[nodiscard]] const Buffer& required_input(
    const ElementInputs& inputs,
    std::string_view pad_name,
    std::string_view element_name)
{
    const auto found = inputs.find(std::string(pad_name));
    if (found == inputs.end() || !found->second) {
        throw std::invalid_argument(std::string(element_name) + " requires connected input pad " + std::string(pad_name));
    }
    return *found->second;
}

[[nodiscard]] std::shared_ptr<leakflow::base::TorchTensorPayload> torch_payload_for(
    const Buffer& buffer,
    std::string_view pad_name)
{
    if (buffer.caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument(std::string(pad_name) + " input must have leakflow/torch-tensor caps");
    }
    auto payload = buffer.payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument(std::string(pad_name) + " input must carry a TorchTensorPayload");
    }
    return payload;
}

[[nodiscard]] std::int64_t integer_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] StringList string_list_property_or(const Element& element, std::string_view name, StringList fallback)
{
    if (const auto value = element.property_as<StringList>(name)) {
        return *value;
    }
    return fallback;
}

void require_truth_tensor(const torch::Tensor& truth)
{
    if (!truth.defined()) {
        throw std::invalid_argument("truth tensor must be defined");
    }
    if (truth.layout() != torch::kStrided) {
        throw std::invalid_argument("truth tensor must use strided layout");
    }
    if (!c10::isIntegralType(truth.scalar_type(), true)) {
        throw std::invalid_argument("truth tensor must be integral");
    }
}

[[nodiscard]] bool unit_indexes_fit_aes_state(const std::vector<std::int64_t>& unit_indexes)
{
    return std::ranges::all_of(unit_indexes, [](std::int64_t value) {
        return value >= 0 && value < 16;
    });
}

[[nodiscard]] std::vector<std::int64_t> truth_values_for_units(
    const torch::Tensor& truth_tensor,
    const std::vector<std::int64_t>& unit_indexes,
    std::int64_t unit_count)
{
    require_truth_tensor(truth_tensor);
    const auto truth = truth_tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    std::vector<std::int64_t> values;
    values.reserve(static_cast<std::size_t>(unit_count));

    if (truth.dim() == 2 && truth.size(1) == 16 && unit_indexes_fit_aes_state(unit_indexes)) {
        for (const auto unit_index : unit_indexes) {
            values.push_back(truth[0][unit_index].item<std::int64_t>());
        }
        return values;
    }

    if (truth.dim() == 1 && truth.size(0) == 16 && unit_indexes_fit_aes_state(unit_indexes)) {
        for (const auto unit_index : unit_indexes) {
            values.push_back(truth[unit_index].item<std::int64_t>());
        }
        return values;
    }

    if (truth.dim() == 1 && truth.size(0) == unit_count) {
        for (std::int64_t unit = 0; unit < unit_count; ++unit) {
            values.push_back(truth[unit].item<std::int64_t>());
        }
        return values;
    }

    if (truth.dim() == 2 && truth.size(0) == 1 && truth.size(1) == unit_count) {
        for (std::int64_t unit = 0; unit < unit_count; ++unit) {
            values.push_back(truth[0][unit].item<std::int64_t>());
        }
        return values;
    }

    throw std::invalid_argument("truth tensor must have shape [U], [1,U], [16], or [N,16]");
}

[[nodiscard]] std::int64_t find_guess_index(const torch::Tensor& guess_values, std::int64_t truth_value)
{
    for (std::int64_t index = 0; index < guess_values.size(0); ++index) {
        if (guess_values[index].item<std::int64_t>() == truth_value) {
            return index;
        }
    }
    throw std::invalid_argument("truth value is not present in attack guess domain");
}

[[nodiscard]] std::string canonical_metric_name(std::string_view value)
{
    std::string canonical;
    canonical.reserve(value.size());
    for (const auto character : value) {
        const auto unsigned_character = static_cast<unsigned char>(character);
        if (character == '-' || std::isspace(unsigned_character)) {
            canonical.push_back('_');
        } else {
            canonical.push_back(static_cast<char>(std::tolower(unsigned_character)));
        }
    }
    return canonical;
}

[[nodiscard]] bool is_allowed_metric(const std::string& metric)
{
    return metric == "margin" || metric == "relative_margin" || metric == "z_score"
        || metric == "robust_z_score" || metric == "top_k_separation";
}

[[nodiscard]] std::vector<std::string> confidence_metrics_for(const Element& element)
{
    auto values = string_list_property_or(
        element,
        "confidence_metrics",
        StringList{"relative_margin", "z_score", "robust_z_score"});

    std::vector<std::string> metrics;
    metrics.reserve(values.size());
    for (const auto& value : values) {
        auto metric = canonical_metric_name(value);
        if (!is_allowed_metric(metric)) {
            throw std::invalid_argument(
                "AttackStats confidence_metrics values must be margin, relative_margin, z_score, "
                "robust_z_score, or top_k_separation");
        }
        if (std::ranges::find(metrics, metric) != metrics.end()) {
            throw std::invalid_argument("AttackStats confidence_metrics must not contain duplicates");
        }
        metrics.push_back(std::move(metric));
    }
    return metrics;
}

[[nodiscard]] double median_of(std::vector<double> values)
{
    if (values.empty()) {
        throw std::invalid_argument("cannot compute median of an empty score vector");
    }
    std::ranges::sort(values);
    const auto middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

[[nodiscard]] double mean_of(const std::vector<double>& values)
{
    double sum = 0.0;
    for (const auto value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

[[nodiscard]] double population_stddev_of(const std::vector<double>& values, double mean)
{
    double square_sum = 0.0;
    for (const auto value : values) {
        const auto centered = value - mean;
        square_sum += centered * centered;
    }
    return std::sqrt(square_sum / static_cast<double>(values.size()));
}

[[nodiscard]] double mad_of(const std::vector<double>& values, double median)
{
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const auto value : values) {
        deviations.push_back(std::abs(value - median));
    }
    return median_of(std::move(deviations));
}

[[nodiscard]] double relative_margin(double score, double next_score)
{
    const auto denominator = std::max(std::abs(score), confidence_epsilon);
    return (score - next_score) / denominator;
}

[[nodiscard]] std::string join_metrics(const std::vector<std::string>& metrics)
{
    std::string output;
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (index != 0) {
            output += ',';
        }
        output += metrics[index];
    }
    return output;
}

void copy_attack_metadata(const Buffer& attack, Buffer& output)
{
    for (const auto& [key, value] : attack.metadata()) {
        if (key.starts_with("attack.") || key.starts_with("payload.leakage.")
            || key.starts_with("payload.crypto.")) {
            output.set_metadata(key, value);
        }
    }
}

} // namespace

ElementDescriptor AttackStats::descriptor()
{
    return {
        .type_name = "AttackStats",
        .klass = "Analyze/SCA/Attack/Stats",
        .purpose = "compute known-key rank and score statistics for an attack score result",
        .input_pads = {
            Pad("scores", PadDirection::Input, Caps(attack_scores_caps_type)),
            Pad("truth", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type),
                PadPresence::Optional),
        },
        .output_pads = {
            Pad("stats", PadDirection::Output, Caps(attack_stats_caps_type)),
        },
        .property_specs = {
            PropertySpec(
                "top_k",
                default_stats_top_k,
                "number of ranked guesses included in top-K diagnostics",
                "",
                IntRangeConstraint{1, 1024},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"stats"}}),
            PropertySpec(
                "confidence_metrics",
                StringList{"relative_margin", "z_score", "robust_z_score"},
                "top-K confidence metric columns shown by summaries",
                "",
                std::monostate{},
                "allowed values: margin, relative_margin, z_score, robust_z_score, top_k_separation",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"stats"}}),
        },
        .keywords = {"attack", "stats", "rank", "pge"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "routing.element", std::string(), "element instance name that produced the attack stats buffer", {"stats"}),
            make_element_metadata_descriptor(
                "payload.stats.id", std::string(), "stats implementation identifier", {attack_stats_id}),
            make_element_metadata_descriptor(
                "attack.stats.has_truth", std::string(),
                "whether a truth input was connected; false means GE/PGE fields are absent",
                {"true", "false"}),
            make_element_metadata_descriptor(
                "attack.stats.rank_base", std::int64_t{},
                "true_rank base; 1 means rank 1 is best (only with truth)", {"1"}),
            make_element_metadata_descriptor(
                "attack.stats.success_count", std::int64_t{},
                "number of units recovered at rank 1 (only with truth)", {"16"}),
            make_element_metadata_descriptor(
                "attack.stats.score_gap", std::string(), "meaning of the legacy score_gap tensor", {"relative_margin"}),
            make_element_metadata_descriptor(
                "attack.stats.top_k", std::int64_t{}, "number of ranked guesses included in top-K diagnostics", {"5"}),
            make_element_metadata_descriptor(
                "attack.stats.confidence_metrics",
                StringList{},
                "selected confidence metrics shown for each top-K guess",
                {"relative_margin,z_score,robust_z_score"}),
        },
    };
}

AttackStats::AttackStats(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> AttackStats::process(std::optional<Buffer>)
{
    throw std::invalid_argument("AttackStats requires a named scores input");
}

std::optional<Buffer> AttackStats::process_inputs(ElementInputs inputs)
{
    const auto& attack_buffer = required_input(inputs, "scores", "AttackStats");
    if (attack_buffer.caps().type() != attack_scores_caps_type) {
        throw std::invalid_argument("scores input must have leakflow/attack-scores caps");
    }
    const auto attack_payload = attack_buffer.payload_as<AttackScoresPayload>();
    if (!attack_payload) {
        throw std::invalid_argument("scores input must carry an AttackScoresPayload");
    }

    const auto unit_count = attack_payload->unit_count();
    const auto guess_count = attack_payload->guess_count();

    // The truth input is optional. Without it, GE/PGE-style diagnostics
    // (true_rank, true_guess, true_score, success) are skipped, while every
    // truth-independent statistic (top-K, margins, separation, ...) is still
    // produced.
    std::optional<std::vector<std::int64_t>> truth_values;
    if (const auto found = inputs.find("truth"); found != inputs.end() && found->second) {
        const auto truth_payload = torch_payload_for(*found->second, "truth");
        truth_values = truth_values_for_units(truth_payload->tensor(), attack_payload->unit_indexes(), unit_count);
    }
    const bool has_truth = truth_values.has_value();

    const auto scores = attack_payload->scores().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto ranking = attack_payload->ranking().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto guess_values = attack_payload->guess_values().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto requested_top_k = integer_property_or(*this, "top_k", default_stats_top_k);
    const auto top_k = std::min(requested_top_k, guess_count);
    const auto confidence_metrics = confidence_metrics_for(*this);

    std::vector<std::int64_t> true_rank_values;
    std::vector<std::int64_t> true_guess_values;
    std::vector<double> true_score_values;
    std::vector<std::int64_t> top1_guess_values;
    std::vector<std::int64_t> top2_guess_values;
    std::vector<double> score_gap_values;
    std::vector<std::uint8_t> success_values;
    std::vector<std::int64_t> topk_guess_values;
    std::vector<double> topk_score_values;
    std::vector<double> topk_margin_values;
    std::vector<double> topk_relative_margin_values;
    std::vector<double> topk_z_score_values;
    std::vector<double> topk_robust_z_score_values;
    std::vector<double> topk_separation_values;
    true_rank_values.reserve(static_cast<std::size_t>(unit_count));
    true_guess_values.reserve(static_cast<std::size_t>(unit_count));
    true_score_values.reserve(static_cast<std::size_t>(unit_count));
    top1_guess_values.reserve(static_cast<std::size_t>(unit_count));
    top2_guess_values.reserve(static_cast<std::size_t>(unit_count));
    score_gap_values.reserve(static_cast<std::size_t>(unit_count));
    success_values.reserve(static_cast<std::size_t>(unit_count));
    topk_guess_values.reserve(static_cast<std::size_t>(unit_count * top_k));
    topk_score_values.reserve(static_cast<std::size_t>(unit_count * top_k));
    topk_margin_values.reserve(static_cast<std::size_t>(unit_count * top_k));
    topk_relative_margin_values.reserve(static_cast<std::size_t>(unit_count * top_k));
    topk_z_score_values.reserve(static_cast<std::size_t>(unit_count * top_k));
    topk_robust_z_score_values.reserve(static_cast<std::size_t>(unit_count * top_k));
    topk_separation_values.reserve(static_cast<std::size_t>(unit_count * top_k));

    for (std::int64_t unit = 0; unit < unit_count; ++unit) {
        std::vector<double> unit_scores;
        unit_scores.reserve(static_cast<std::size_t>(guess_count));
        for (std::int64_t guess = 0; guess < guess_count; ++guess) {
            unit_scores.push_back(scores[unit][guess].item<double>());
        }
        const auto mean = mean_of(unit_scores);
        const auto stddev = population_stddev_of(unit_scores, mean);
        const auto median = median_of(unit_scores);
        const auto mad = mad_of(unit_scores, median);
        const auto outside_topk_score = guess_count > top_k
            ? scores[unit][ranking[unit][top_k].item<std::int64_t>()].item<double>()
            : std::numeric_limits<double>::quiet_NaN();

        const auto top1_index = ranking[unit][0].item<std::int64_t>();
        const auto top2_index = ranking[unit][std::min<std::int64_t>(1, guess_count - 1)].item<std::int64_t>();
        const auto top1_score = scores[unit][top1_index].item<double>();
        const auto top2_score = scores[unit][top2_index].item<double>();
        const auto top1_relative_margin = guess_count > 1 ? relative_margin(top1_score, top2_score) : 0.0;

        top1_guess_values.push_back(guess_values[top1_index].item<std::int64_t>());
        top2_guess_values.push_back(guess_values[top2_index].item<std::int64_t>());
        score_gap_values.push_back(top1_relative_margin);

        if (has_truth) {
            const auto truth_value = (*truth_values)[static_cast<std::size_t>(unit)];
            const auto truth_guess_index = find_guess_index(guess_values, truth_value);
            std::int64_t rank_position = -1;
            for (std::int64_t rank = 0; rank < guess_count; ++rank) {
                if (ranking[unit][rank].item<std::int64_t>() == truth_guess_index) {
                    rank_position = rank;
                    break;
                }
            }
            if (rank_position < 0) {
                throw std::invalid_argument("CPA ranking did not contain the truth guess index");
            }

            true_rank_values.push_back(rank_position + 1);
            true_guess_values.push_back(truth_value);
            true_score_values.push_back(scores[unit][truth_guess_index].item<double>());
            success_values.push_back(rank_position == 0 ? std::uint8_t{1} : std::uint8_t{0});
        }

        for (std::int64_t top_index = 0; top_index < top_k; ++top_index) {
            const auto guess_index = ranking[unit][top_index].item<std::int64_t>();
            const auto score = scores[unit][guess_index].item<double>();
            const auto next_score = top_index + 1 < guess_count
                ? scores[unit][ranking[unit][top_index + 1].item<std::int64_t>()].item<double>()
                : score;
            const auto margin = top_index + 1 < guess_count ? score - next_score : 0.0;
            const auto separation = std::isfinite(outside_topk_score) ? score - outside_topk_score : 0.0;

            topk_guess_values.push_back(guess_values[guess_index].item<std::int64_t>());
            topk_score_values.push_back(score);
            topk_margin_values.push_back(margin);
            topk_relative_margin_values.push_back(top_index + 1 < guess_count ? relative_margin(score, next_score) : 0.0);
            topk_z_score_values.push_back(stddev > confidence_epsilon ? (score - mean) / stddev : 0.0);
            topk_robust_z_score_values.push_back(
                mad > confidence_epsilon ? robust_z_scale * (score - median) / mad : 0.0);
            topk_separation_values.push_back(separation);
        }
    }

    std::optional<torch::Tensor> true_rank_tensor;
    std::optional<torch::Tensor> true_guess_tensor;
    std::optional<torch::Tensor> true_score_tensor;
    std::optional<torch::Tensor> success_tensor;
    if (has_truth) {
        true_rank_tensor = torch::tensor(true_rank_values, torch::TensorOptions().dtype(torch::kInt64));
        true_guess_tensor = torch::tensor(true_guess_values, torch::TensorOptions().dtype(torch::kInt64));
        true_score_tensor = torch::tensor(true_score_values, torch::TensorOptions().dtype(torch::kFloat64));
        success_tensor =
            torch::tensor(success_values, torch::TensorOptions().dtype(torch::kUInt8)).to(torch::kBool);
    }

    auto stats_payload = std::make_shared<AttackStatsPayload>(
        std::move(true_rank_tensor),
        std::move(true_guess_tensor),
        std::move(true_score_tensor),
        torch::tensor(top1_guess_values, torch::TensorOptions().dtype(torch::kInt64)),
        torch::tensor(top2_guess_values, torch::TensorOptions().dtype(torch::kInt64)),
        torch::tensor(score_gap_values, torch::TensorOptions().dtype(torch::kFloat64)),
        std::move(success_tensor),
        attack_payload->best_channel().to(torch::kCPU).to(torch::kInt64).contiguous(),
        attack_payload->best_sample().to(torch::kCPU).to(torch::kInt64).contiguous(),
        torch::tensor(topk_guess_values, torch::TensorOptions().dtype(torch::kInt64)).reshape({unit_count, top_k}),
        torch::tensor(topk_score_values, torch::TensorOptions().dtype(torch::kFloat64)).reshape({unit_count, top_k}),
        torch::tensor(topk_margin_values, torch::TensorOptions().dtype(torch::kFloat64)).reshape({unit_count, top_k}),
        torch::tensor(topk_relative_margin_values, torch::TensorOptions().dtype(torch::kFloat64)).reshape({unit_count, top_k}),
        torch::tensor(topk_z_score_values, torch::TensorOptions().dtype(torch::kFloat64)).reshape({unit_count, top_k}),
        torch::tensor(topk_robust_z_score_values, torch::TensorOptions().dtype(torch::kFloat64)).reshape({unit_count, top_k}),
        torch::tensor(topk_separation_values, torch::TensorOptions().dtype(torch::kFloat64)).reshape({unit_count, top_k}),
        attack_payload->unit_indexes(),
        attack_payload->channel_names(),
        confidence_metrics);

    Buffer output{Caps(attack_stats_caps_type)};
    forward_metadata(inputs, profile_for_klass(element_kclass()), output, name());
    copy_attack_metadata(attack_buffer, output);
    output.set_metadata("routing.element", name());
    output.set_metadata("payload.stats.id", attack_stats_id);
    output.set_metadata("attack.stats.has_truth", has_truth ? "true" : "false");
    output.set_metadata("attack.stats.score_gap", "relative_margin");
    output.set_metadata("attack.stats.top_k", std::to_string(top_k));
    output.set_metadata("attack.stats.confidence_metrics", join_metrics(confidence_metrics));
    if (has_truth) {
        output.set_metadata("attack.stats.rank_base", "1");
        output.set_metadata("attack.stats.success_count",
            std::to_string(stats_payload->success()->to(torch::kCPU).to(torch::kInt64).sum().item<std::int64_t>()));
    }
    output.set_payload(std::move(stats_payload));

    auto record = make_log_record(log::LogLevel::Debug, "element", "computed attack stats");
    record.fields.emplace("payload.stats.id", attack_stats_id);
    record.fields.emplace("has_truth", has_truth ? "true" : "false");
    if (has_truth) {
        record.fields.emplace("success_count", output.metadata("attack.stats.success_count"));
    }
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::crypto
