#include "leakflow/plugins/crypto/cpa_attack_stats_to_plot_annotations.hpp"

#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/cpa_attack_payload.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::crypto {
namespace {

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] std::int64_t integer_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] std::string format_value(double value, std::string_view value_format, std::int64_t precision)
{
    std::ostringstream output;
    if (value_format == "fixed") {
        output << std::fixed;
    } else if (value_format == "scientific") {
        output << std::scientific;
    } else {
        throw std::invalid_argument("CpaAttackStatsToPlotAnnotations value_format must be fixed or scientific");
    }
    output << std::setprecision(static_cast<int>(precision)) << value;
    return output.str();
}

[[nodiscard]] std::string unit_label(
    const std::vector<std::int64_t>& unit_indexes,
    std::int64_t unit)
{
    if (unit >= 0 && static_cast<std::size_t>(unit) < unit_indexes.size()) {
        return std::to_string(unit_indexes[static_cast<std::size_t>(unit)]);
    }
    return std::to_string(unit);
}

[[nodiscard]] std::string channel_label(
    const std::vector<std::string>& channel_names,
    std::int64_t channel_index)
{
    if (channel_index >= 0 && static_cast<std::size_t>(channel_index) < channel_names.size()) {
        return channel_names[static_cast<std::size_t>(channel_index)];
    }
    return "channel_" + std::to_string(channel_index);
}

[[nodiscard]] std::shared_ptr<CpaAttackStatsPayload> stats_payload_for(const Buffer& input)
{
    if (input.caps().type() != cpa_attack_stats_caps_type) {
        throw std::invalid_argument("CpaAttackStatsToPlotAnnotations requires leakflow/cpa-attack-stats input caps");
    }
    auto payload = input.payload_as<CpaAttackStatsPayload>();
    if (!payload) {
        throw std::invalid_argument("CpaAttackStatsToPlotAnnotations requires a CpaAttackStatsPayload");
    }
    return payload;
}

[[nodiscard]] std::vector<leakflow::base::PlotAnnotation> annotations_from(
    const CpaAttackStatsPayload& payload,
    std::string_view kind,
    std::string_view value_format,
    std::int64_t precision,
    std::int64_t max_units)
{
    if (precision < 0 || precision > 12) {
        throw std::invalid_argument("CpaAttackStatsToPlotAnnotations precision must be between 0 and 12");
    }
    if (max_units <= 0) {
        throw std::invalid_argument("CpaAttackStatsToPlotAnnotations max_units must be positive");
    }

    const auto unit_count = std::min(payload.unit_count(), max_units);
    const auto success_tensor = payload.success().to(torch::kCPU).to(torch::kBool).contiguous();
    const auto true_rank_tensor = payload.true_rank().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto true_guess_tensor = payload.true_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto true_score_tensor = payload.true_score().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto top1_guess_tensor = payload.top1_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto topk_score_tensor = payload.topk_score().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto relative_margin_tensor = payload.score_gap().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto best_channel_tensor = payload.best_channel().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto best_sample_tensor = payload.best_sample().to(torch::kCPU).to(torch::kInt64).contiguous();

    std::vector<leakflow::base::PlotAnnotation> annotations;
    annotations.reserve(static_cast<std::size_t>(unit_count));
    for (std::int64_t unit = 0; unit < unit_count; ++unit) {
        const auto success = success_tensor[unit].item<bool>();
        const auto true_rank = true_rank_tensor[unit].item<std::int64_t>();
        const auto true_guess = true_guess_tensor[unit].item<std::int64_t>();
        const auto true_score = true_score_tensor[unit].item<double>();
        const auto top1_guess = top1_guess_tensor[unit].item<std::int64_t>();
        const auto top1_score = topk_score_tensor[unit][0].item<double>();
        const auto relative_margin = relative_margin_tensor[unit].item<double>();
        const auto best_channel = best_channel_tensor[unit].item<std::int64_t>();
        const auto best_sample = best_sample_tensor[unit].item<std::int64_t>();
        const auto attack_unit = unit_label(payload.unit_indexes(), unit);
        const auto channel = channel_label(payload.channel_names(), best_channel);
        const auto score_text = format_value(top1_score, value_format, precision);
        const auto true_score_text = format_value(true_score, value_format, precision);
        const auto relative_margin_text = format_value(relative_margin, value_format, precision);
        const auto norm_magnitude = std::clamp(std::abs(top1_score), 0.0, 1.0);
        const auto signed_norm = success ? norm_magnitude : -norm_magnitude;
        const auto label = "unit_" + attack_unit + "." + channel;

        leakflow::base::PlotAnnotationFields fields;
        fields.emplace_back("attack unit", attack_unit);
        fields.emplace_back("success", success ? "true" : "false");
        fields.emplace_back("guess", std::to_string(top1_guess));
        fields.emplace_back("channel", channel);
        fields.emplace_back("score", score_text);
        fields.emplace_back("true rank", std::to_string(true_rank));
        fields.emplace_back("true guess", std::to_string(true_guess));
        fields.emplace_back("true score", true_score_text);
        fields.emplace_back("relative margin", relative_margin_text);
        fields.emplace_back("score_gap", relative_margin_text);
        if (!success) {
            fields.emplace_back(
                "correct key",
                "guess=" + std::to_string(true_guess) + " rank=" + std::to_string(true_rank)
                    + " score=" + true_score_text);
        }

        auto text = label + ": guess=" + std::to_string(top1_guess)
            + " score=" + score_text
            + " success=" + std::string(success ? "true" : "false")
            + " true_guess=" + std::to_string(true_guess)
            + " true_rank=" + std::to_string(true_rank)
            + " true_score=" + true_score_text;
        if (!success) {
            text += " correct_key=guess=" + std::to_string(true_guess)
                + ",rank=" + std::to_string(true_rank)
                + ",score=" + true_score_text;
        }

        annotations.push_back(leakflow::base::PlotAnnotation{
            .sample_index = best_sample,
            .norm_value = signed_norm,
            .fields = std::move(fields),
            .label = label,
            .text = std::move(text),
            .kind = std::string(kind),
            .target_index = unit,
        });
    }

    return annotations;
}

} // namespace

ElementDescriptor CpaAttackStatsToPlotAnnotations::descriptor()
{
    return {
        .type_name = "CpaAttackStatsToPlotAnnotations",
        .klass = "Convert/SCA/Plot/Annotations",
        .purpose = "convert CPA known-key statistics into generic plot annotations",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(cpa_attack_stats_caps_type)),
        },
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(leakflow::base::plot_annotation_caps_type)),
        },
        .property_specs = {
            PropertySpec(
                "kind",
                std::string("cpa"),
                "plot annotation kind",
                "",
                std::monostate{},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"}}),
            PropertySpec(
                "value_format",
                std::string("fixed"),
                "annotation value format",
                "",
                StringEnumConstraint{{"fixed", "scientific"}},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"}}),
            PropertySpec(
                "precision",
                std::int64_t{3},
                "annotation displayed value precision",
                "",
                IntRangeConstraint{0, 12},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"}}),
            PropertySpec(
                "max_units",
                std::int64_t{64},
                "maximum attack units to annotate",
                "",
                IntRangeConstraint{1, 1024},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"}}),
        },
        .keywords = {"cpa", "attack", "stats", "plot", "annotation", cpa_attack_stats_to_plot_annotations_id},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.conversion.id",
                std::string(),
                "conversion implementation identifier",
                {cpa_attack_stats_to_plot_annotations_id}),
            make_element_metadata_descriptor(
                "payload.conversion.element",
                std::string(),
                "element instance name that performed the conversion",
                {"ann"}),
            make_element_metadata_descriptor(
                "payload.annotation.kind",
                std::string(),
                "annotation kind stamped onto plot annotations",
                {"cpa"}),
            make_element_metadata_descriptor(
                "payload.annotation.count",
                std::int64_t{},
                "number of plot annotations emitted",
                {"16"}),
            make_element_metadata_descriptor(
                "payload.annotation.success_source",
                std::string(),
                "source used to sign annotations by success or failure",
                {"stats"}),
        },
    };
}

CpaAttackStatsToPlotAnnotations::CpaAttackStatsToPlotAnnotations(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> CpaAttackStatsToPlotAnnotations::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("CpaAttackStatsToPlotAnnotations requires an input buffer");
    }
    const auto payload = stats_payload_for(*input);

    const auto kind = string_property_or(*this, "kind", "cpa");
    const auto value_format = string_property_or(*this, "value_format", "fixed");
    const auto precision = integer_property_or(*this, "precision", 3);
    const auto max_units = integer_property_or(*this, "max_units", 64);
    auto annotation_payload = leakflow::base::PlotAnnotationPayload(
        annotations_from(*payload, kind, value_format, precision, max_units));

    Buffer output{annotation_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "sink", name());
    output.set_metadata("payload.conversion.id", cpa_attack_stats_to_plot_annotations_id);
    output.set_metadata("payload.conversion.element", name());
    output.set_metadata("payload.annotation.kind", kind);
    output.set_metadata("payload.annotation.count", std::to_string(annotation_payload.annotation_count()));
    output.set_metadata("payload.annotation.success_source", "stats");
    output.set_payload(std::make_shared<leakflow::base::PlotAnnotationPayload>(std::move(annotation_payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "converted CPA attack stats to plot annotations");
    record.fields.emplace("payload.conversion.id", cpa_attack_stats_to_plot_annotations_id);
    record.fields.emplace("annotations", output.metadata("payload.annotation.count"));
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::crypto
