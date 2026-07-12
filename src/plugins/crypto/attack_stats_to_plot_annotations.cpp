#include "leakflow/plugins/crypto/attack_stats_to_plot_annotations.hpp"

#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/attack_payload.hpp"

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
        throw std::invalid_argument("AttackStatsToPlotAnnotations value_format must be fixed or scientific");
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

[[nodiscard]] std::shared_ptr<AttackStatsPayload> stats_payload_for(const Buffer& input)
{
    if (input.caps().type() != attack_stats_caps_type) {
        throw std::invalid_argument("AttackStatsToPlotAnnotations requires leakflow/attack-stats input caps");
    }
    auto payload = input.payload_as<AttackStatsPayload>();
    if (!payload) {
        throw std::invalid_argument("AttackStatsToPlotAnnotations requires an AttackStatsPayload");
    }
    return payload;
}

void copy_attack_metadata(const Buffer& input, Buffer& output)
{
    for (const auto& [key, value] : input.metadata()) {
        if (key.starts_with("attack.")) {
            output.set_metadata(key, value);
        }
    }
}

[[nodiscard]] std::vector<leakflow::base::PlotAnnotation> annotations_from(
    const AttackStatsPayload& payload,
    std::string_view kind,
    std::string_view value_format,
    std::int64_t precision,
    std::int64_t max_units)
{
    if (precision < 0 || precision > 12) {
        throw std::invalid_argument("AttackStatsToPlotAnnotations precision must be between 0 and 12");
    }
    if (max_units <= 0) {
        throw std::invalid_argument("AttackStatsToPlotAnnotations max_units must be positive");
    }

    const auto unit_count = std::min(payload.unit_count(), max_units);
    const bool has_truth = payload.has_truth();
    torch::Tensor success_tensor;
    torch::Tensor true_rank_tensor;
    torch::Tensor true_guess_tensor;
    torch::Tensor true_score_tensor;
    if (has_truth) {
        success_tensor = payload.success()->to(torch::kCPU).to(torch::kBool).contiguous();
        true_rank_tensor = payload.true_rank()->to(torch::kCPU).to(torch::kInt64).contiguous();
        true_guess_tensor = payload.true_guess()->to(torch::kCPU).to(torch::kInt64).contiguous();
        true_score_tensor = payload.true_score()->to(torch::kCPU).to(torch::kFloat64).contiguous();
    }
    const auto top1_guess_tensor = payload.top1_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto topk_score_tensor = payload.topk_score().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto relative_margin_tensor = payload.score_gap().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto best_channel_tensor = payload.best_channel().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto best_sample_tensor = payload.best_sample().to(torch::kCPU).to(torch::kInt64).contiguous();

    std::vector<leakflow::base::PlotAnnotation> annotations;
    annotations.reserve(static_cast<std::size_t>(unit_count));
    for (std::int64_t unit = 0; unit < unit_count; ++unit) {
        const auto success = has_truth && success_tensor[unit].item<bool>();
        const auto top1_guess = top1_guess_tensor[unit].item<std::int64_t>();
        const auto top1_score = topk_score_tensor[unit][0].item<double>();
        const auto relative_margin = relative_margin_tensor[unit].item<double>();
        const auto best_channel = best_channel_tensor[unit].item<std::int64_t>();
        const auto best_sample = best_sample_tensor[unit].item<std::int64_t>();
        const auto attack_unit = unit_label(payload.unit_indexes(), unit);
        const auto channel = channel_label(payload.channel_names(), best_channel);
        const auto score_text = format_value(top1_score, value_format, precision);
        const auto relative_margin_text = format_value(relative_margin, value_format, precision);
        // Annotation height is always the (positive) score magnitude; success/failure
        // is encoded by the marker shape instead of the sign, so failed units are not
        // pushed below the axis. circle=unknown (no truth), square=success, x=failure.
        const auto norm_magnitude = std::clamp(std::abs(top1_score), 0.0, 1.0);
        const std::string marker = !has_truth ? "circle" : (success ? "square" : "x");
        const auto label = "unit " + attack_unit + " [" + channel + "]";

        leakflow::base::PlotAnnotationFields fields;
        fields.emplace_back("attack unit", attack_unit);
        if (has_truth) {
            fields.emplace_back("success", success ? "true" : "false");
        }
        fields.emplace_back("guess", std::to_string(top1_guess));
        fields.emplace_back("channel", channel);
        fields.emplace_back("score", score_text);

        std::int64_t true_rank = 0;
        std::int64_t true_guess = 0;
        std::string true_score_text;
        if (has_truth) {
            true_rank = true_rank_tensor[unit].item<std::int64_t>();
            true_guess = true_guess_tensor[unit].item<std::int64_t>();
            true_score_text = format_value(true_score_tensor[unit].item<double>(), value_format, precision);
            fields.emplace_back("true rank", std::to_string(true_rank));
            fields.emplace_back("true guess", std::to_string(true_guess));
            fields.emplace_back("true score", true_score_text);
        }
        fields.emplace_back("relative margin", relative_margin_text);
        fields.emplace_back("score_gap", relative_margin_text);
        if (has_truth && !success) {
            fields.emplace_back(
                "correct key",
                "guess=" + std::to_string(true_guess) + " rank=" + std::to_string(true_rank)
                    + " score=" + true_score_text);
        }

        auto text = label + ": guess=" + std::to_string(top1_guess) + " score=" + score_text;
        if (has_truth) {
            text += " success=" + std::string(success ? "true" : "false")
                + " true_guess=" + std::to_string(true_guess)
                + " true_rank=" + std::to_string(true_rank)
                + " true_score=" + true_score_text;
            if (!success) {
                text += " correct_key=guess=" + std::to_string(true_guess)
                    + ",rank=" + std::to_string(true_rank)
                    + ",score=" + true_score_text;
            }
        }

        annotations.push_back(leakflow::base::PlotAnnotation{
            .sample_index = best_sample,
            .norm_value = norm_magnitude,
            .fields = std::move(fields),
            .label = label,
            .text = std::move(text),
            .kind = std::string(kind),
            .target_index = unit,
            .marker = marker,
        });
    }

    return annotations;
}

} // namespace

ElementDescriptor AttackStatsToPlotAnnotations::descriptor()
{
    return {
        .type_name = "AttackStatsToPlotAnnotations",
        .klass = "Convert/PlotAnnotation/AttackStats",
        .purpose = "convert attack known-key statistics into generic plot annotations",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(attack_stats_caps_type)),
        },
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(leakflow::base::plot_annotation_caps_type)),
        },
        .property_specs = {
            PropertySpec(
                "kind",
                std::string("attack"),
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
        .keywords = {"attack", "stats", "plot", "annotation", attack_stats_to_plot_annotations_id},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.conversion.id",
                std::string(),
                "conversion implementation identifier",
                {attack_stats_to_plot_annotations_id}),
            make_element_metadata_descriptor(
                "payload.conversion.element",
                std::string(),
                "element instance name that performed the conversion",
                {"ann"}),
            make_element_metadata_descriptor(
                "payload.annotation.kind",
                std::string(),
                "annotation kind stamped onto plot annotations",
                {"attack"}),
            make_element_metadata_descriptor(
                "payload.annotation.count",
                std::int64_t{},
                "number of plot annotations emitted",
                {"16"}),
            make_element_metadata_descriptor(
                "payload.annotation.success_source",
                std::string(),
                "source of the success marker (stats when truth is present: square=success, "
                "x=failure; none -> circle)",
                {"stats", "none"}),
            make_element_metadata_descriptor(
                "payload.layout", std::string(), "semantic payload layout",
                {"annotation/[sample_index,value?,norm_value?,fields,label,text,kind,target_index?,marker]"}),
        },
    };
}

AttackStatsToPlotAnnotations::AttackStatsToPlotAnnotations(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> AttackStatsToPlotAnnotations::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("AttackStatsToPlotAnnotations requires an input buffer");
    }
    const auto payload = stats_payload_for(*input);

    const auto kind = string_property_or(*this, "kind", "attack");
    const auto value_format = string_property_or(*this, "value_format", "fixed");
    const auto precision = integer_property_or(*this, "precision", 3);
    const auto max_units = integer_property_or(*this, "max_units", 64);
    auto annotation_payload = leakflow::base::PlotAnnotationPayload(
        annotations_from(*payload, kind, value_format, precision, max_units));

    Buffer output{annotation_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "sink", name());
    copy_attack_metadata(*input, output);
    output.set_metadata("payload.conversion.id", attack_stats_to_plot_annotations_id);
    output.set_metadata("payload.conversion.element", name());
    output.set_metadata("payload.annotation.kind", kind);
    output.set_metadata("payload.annotation.count", std::to_string(annotation_payload.annotation_count()));
    output.set_metadata("payload.annotation.success_source", payload->has_truth() ? "stats" : "none");
    output.set_payload(std::make_shared<leakflow::base::PlotAnnotationPayload>(std::move(annotation_payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "converted attack stats to plot annotations");
    record.fields.emplace("payload.conversion.id", attack_stats_to_plot_annotations_id);
    record.fields.emplace("annotations", output.metadata("payload.annotation.count"));
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::crypto
