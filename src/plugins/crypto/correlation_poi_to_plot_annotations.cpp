#include "leakflow/plugins/crypto/correlation_poi_to_plot_annotations.hpp"

#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"
#include "leakflow/plugins/crypto/poi_select.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

[[nodiscard]] std::string trim_to_string(std::string_view text)
{
    const auto begin = text.find_first_not_of(" \t\n\r");
    if (begin == std::string_view::npos) {
        return {};
    }

    const auto end = text.find_last_not_of(" \t\n\r");
    return std::string(text.substr(begin, end - begin + 1));
}

[[nodiscard]] std::vector<std::string> split_comma_list(std::string_view text)
{
    std::vector<std::string> values;
    std::size_t begin = 0;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        if (index == text.size() || text[index] == ',') {
            values.push_back(trim_to_string(text.substr(begin, index - begin)));
            begin = index + 1;
        }
    }
    return values;
}

[[nodiscard]] std::vector<std::string> target_channels_from_metadata(const Buffer& buffer)
{
    if (!buffer.has_metadata("payload.leakage.channels")) {
        return {};
    }

    return split_comma_list(buffer.metadata("payload.leakage.channels"));
}

[[nodiscard]] std::string format_value(double value, std::string_view value_format, std::int64_t precision)
{
    std::ostringstream output;
    if (value_format == "fixed") {
        output << std::fixed;
    } else if (value_format == "scientific") {
        output << std::scientific;
    } else {
        throw std::invalid_argument("CorrelationPoiToPlotAnnotations value_format must be fixed or scientific");
    }
    output << std::setprecision(static_cast<int>(precision)) << value;
    return output.str();
}

[[nodiscard]] std::string annotation_text_for(
    std::string_view label,
    std::string_view score_name,
    std::string_view value_text)
{
    if (score_name == "correlation") {
        return std::string(label) + ": " + std::string(value_text);
    }
    return std::string(label) + ": " + std::string(score_name) + "=" + std::string(value_text);
}

[[nodiscard]] double pearson_correlation_norm_value(double value)
{
    if (!std::isfinite(value) || value < -1.0 || value > 1.0) {
        throw std::invalid_argument(
            "CorrelationPoiToPlotAnnotations Pearson correlation value must be finite and between -1 and 1");
    }
    return value;
}

[[nodiscard]] double normalized_score_for(const Buffer& input, double value)
{
    if (input.metadata_or("payload.poi.method", "") == pearson_poi_method_id) {
        return pearson_correlation_norm_value(value);
    }

    return std::clamp(value, -1.0, 1.0);
}

[[nodiscard]] std::string fallback_channel_label(std::int64_t channel_index)
{
    return "channel_" + std::to_string(channel_index);
}

[[nodiscard]] std::string channel_label_for(
    const std::vector<std::string>& channels,
    std::int64_t channel_index)
{
    if (channel_index >= 0 && static_cast<std::size_t>(channel_index) < channels.size()
        && !channels.at(static_cast<std::size_t>(channel_index)).empty()) {
        return channels.at(static_cast<std::size_t>(channel_index));
    }

    return fallback_channel_label(channel_index);
}

[[nodiscard]] std::string annotation_label_for(std::uint16_t unit, std::string_view channel)
{
    return "unit_" + std::to_string(unit) + "." + std::string(channel);
}

[[nodiscard]] std::string annotation_field_label_for(std::uint16_t unit, std::string_view channel)
{
    return std::string(channel) + "[" + std::to_string(unit) + "]";
}

[[nodiscard]] leakflow::base::PlotAnnotationFields annotation_fields_for(
    std::uint16_t unit,
    std::string_view channel,
    std::string_view score_name,
    std::string_view value_text)
{
    leakflow::base::PlotAnnotationFields fields;
    fields.emplace_back("label", annotation_field_label_for(unit, channel));
    fields.emplace_back("unit", std::to_string(unit));
    fields.emplace_back("target", std::string(channel));
    fields.emplace_back(std::string(score_name), std::string(value_text));
    return fields;
}

[[nodiscard]] std::vector<leakflow::base::PlotAnnotation> annotations_from(
    const Buffer& input,
    const CorrelationPoiPayload& payload,
    std::string_view kind,
    std::string_view value_format,
    std::int64_t precision)
{
    if (precision < 0 || precision > 12) {
        throw std::invalid_argument("CorrelationPoiToPlotAnnotations precision must be between 0 and 12");
    }

    std::vector<leakflow::base::PlotAnnotation> annotations;
    const auto channels = target_channels_from_metadata(input);
    for (std::size_t group_ordinal = 0; group_ordinal < payload.results().size(); ++group_ordinal) {
        const auto& result = payload.results().at(group_ordinal);
        auto pairs = result.result.to(torch::kCPU).to(torch::kFloat64).contiguous();
        const auto channel_count = pairs.size(0);
        const auto rows = pairs.size(1);
        if (!channels.empty() && channels.size() != static_cast<std::size_t>(channel_count)) {
            throw std::invalid_argument(
                "CorrelationPoiToPlotAnnotations leakage channel metadata must match payload channel axis");
        }
        annotations.reserve(
            annotations.size() + static_cast<std::size_t>(channel_count * rows));

        for (auto channel_index = std::int64_t{0}; channel_index < channel_count; ++channel_index) {
            const auto channel = channel_label_for(channels, channel_index);
            const auto label = annotation_label_for(result.unit, channel);
            const auto target_index =
                static_cast<std::int64_t>(group_ordinal) * channel_count + channel_index;
            for (auto row = std::int64_t{0}; row < rows; ++row) {
                const auto sample_index = static_cast<std::int64_t>(pairs[channel_index][row][0].item<double>());
                const auto value = pairs[channel_index][row][1].item<double>();
                const auto value_text = format_value(value, value_format, precision);
                annotations.push_back(leakflow::base::PlotAnnotation{
                    .sample_index = sample_index,
                    .norm_value = normalized_score_for(input, value),
                    .fields = annotation_fields_for(
                        result.unit,
                        channel,
                        payload.score_name(),
                        value_text),
                    .label = label,
                    .text = annotation_text_for(label, payload.score_name(), value_text),
                    .kind = std::string(kind),
                    .target_index = target_index,
                });
            }
        }
    }

    return annotations;
}

} // namespace

ElementDescriptor CorrelationPoiToPlotAnnotations::descriptor()
{
    return {
        .type_name = "CorrelationPoiToPlotAnnotations",
        .klass = "Convert/PlotAnnotation/PoI",
        .purpose = "convert correlation PoI selections into generic plot annotations",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(correlation_poi_caps_type)),
        },
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(leakflow::base::plot_annotation_caps_type)),
        },
        .property_specs = {
            PropertySpec("kind", std::string("poi"), "plot annotation kind",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"},
                }),
            PropertySpec(
                "value_format",
                std::string("fixed"),
                "annotation value format",
                "",
                StringEnumConstraint{{"fixed", "scientific"}},
                "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"},
                }),
            PropertySpec(
                "precision",
                std::int64_t{3},
                "annotation displayed value precision",
                "",
                IntRangeConstraint{0, 12},
                "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"},
                }),
        },
        .keywords = {"poi", "plot", "annotation", "correlation", correlation_poi_to_plot_annotations_id},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.conversion.id",
                std::string(),
                "conversion implementation identifier",
                {correlation_poi_to_plot_annotations_id}),
            make_element_metadata_descriptor(
                "payload.conversion.element",
                std::string(),
                "element instance name that performed the conversion",
                {"ann"}),
            make_element_metadata_descriptor(
                "payload.annotation.kind",
                std::string(),
                "annotation kind stamped onto plot annotations",
                {"poi"}),
            make_element_metadata_descriptor(
                "payload.annotation.count",
                std::int64_t{},
                "number of plot annotations emitted",
                {"50"}),
        },
    };
}

CorrelationPoiToPlotAnnotations::CorrelationPoiToPlotAnnotations(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> CorrelationPoiToPlotAnnotations::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("CorrelationPoiToPlotAnnotations requires an input buffer");
    }
    if (input->caps().type() != correlation_poi_caps_type) {
        throw std::invalid_argument("CorrelationPoiToPlotAnnotations requires leakflow/correlation-poi input caps");
    }

    const auto payload = input->payload_as<CorrelationPoiPayload>();
    if (!payload) {
        throw std::invalid_argument("CorrelationPoiToPlotAnnotations requires a CorrelationPoiPayload");
    }

    const auto kind = string_property_or(*this, "kind", "poi");
    const auto value_format = string_property_or(*this, "value_format", "fixed");
    const auto precision = integer_property_or(*this, "precision", 3);
    auto annotation_payload = leakflow::base::PlotAnnotationPayload(
        annotations_from(*input, *payload, kind, value_format, precision));

    Buffer output{annotation_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "sink", name());
    output.set_metadata("payload.conversion.id", correlation_poi_to_plot_annotations_id);
    output.set_metadata("payload.conversion.element", name());
    output.set_metadata("payload.annotation.kind", kind);
    output.set_metadata("payload.annotation.count", std::to_string(annotation_payload.annotation_count()));
    output.set_payload(std::make_shared<leakflow::base::PlotAnnotationPayload>(std::move(annotation_payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "converted correlation PoIs to plot annotations");
    record.fields.emplace("payload.conversion.id", correlation_poi_to_plot_annotations_id);
    record.fields.emplace("annotations", output.metadata("payload.annotation.count"));
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::crypto
