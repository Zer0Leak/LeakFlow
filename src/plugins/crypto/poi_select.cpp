#include "leakflow/plugins/crypto/poi_select.hpp"

#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/correlation_payload.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::crypto {
namespace {

enum class RankBy {
    Abs,
    Positive,
    Negative,
};

[[nodiscard]] std::string lower_string(std::string_view text)
{
    std::string lowered;
    lowered.reserve(text.size());
    for (const auto character : text) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    return lowered;
}

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] IntList int_list_property_or(const Element& element, std::string_view name, IntList fallback)
{
    if (const auto value = element.property_as<IntList>(name)) {
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

[[nodiscard]] RankBy rank_by_for(std::string_view text)
{
    const auto lowered = lower_string(text);
    if (lowered == "abs") {
        return RankBy::Abs;
    }
    if (lowered == "positive") {
        return RankBy::Positive;
    }
    if (lowered == "negative") {
        return RankBy::Negative;
    }
    throw std::invalid_argument("PoiSelect rank_by values must be abs, positive, or negative");
}

// rank_by is applied per leakage feature/channel (for example HW(m), HW(y)). One value
// applies to every channel, or one value per channel. Indexed by channel.
[[nodiscard]] std::vector<RankBy> rank_modes_for(const Element& element, std::int64_t channel_count)
{
    auto values = string_list_property_or(element, "rank_by", StringList{"abs"});
    if (values.empty()) {
        throw std::invalid_argument("PoiSelect rank_by cannot be empty");
    }
    std::vector<RankBy> modes;
    if (values.size() == 1) {
        modes.assign(static_cast<std::size_t>(channel_count), rank_by_for(values.front()));
        return modes;
    }
    if (values.size() != static_cast<std::size_t>(channel_count)) {
        throw std::invalid_argument("PoiSelect rank_by must have length 1 or the number of leakage channels");
    }
    modes.reserve(values.size());
    for (const auto& value : values) {
        modes.push_back(rank_by_for(value));
    }
    return modes;
}

[[nodiscard]] std::vector<std::int64_t> per_unit_top_k_for(const Element& element, std::int64_t unit_count)
{
    auto values = int_list_property_or(element, "top_k", IntList{10});
    if (values.empty()) {
        throw std::invalid_argument("PoiSelect top_k cannot be empty");
    }
    if (values.size() == 1) {
        return std::vector<std::int64_t>(static_cast<std::size_t>(unit_count), values.front());
    }
    if (values.size() != static_cast<std::size_t>(unit_count)) {
        throw std::invalid_argument("PoiSelect top_k must have length 1 or unit count");
    }
    return values;
}

void validate_top_k(std::int64_t top_k, std::int64_t feature_count)
{
    if (top_k <= 0) {
        throw std::invalid_argument("PoiSelect top_k values must be positive");
    }
    if (top_k > feature_count) {
        throw std::invalid_argument("PoiSelect top_k values must not exceed feature count");
    }
}

[[nodiscard]] torch::Tensor contribution_for(const torch::Tensor& correlations, RankBy rank_by)
{
    switch (rank_by) {
    case RankBy::Abs:
        return correlations.abs();
    case RankBy::Positive:
        return correlations.clamp_min(0);
    case RankBy::Negative:
        return (-correlations).clamp_min(0);
    }
    throw std::invalid_argument("unsupported PoiSelect rank_by");
}

[[nodiscard]] torch::Tensor pairs_from_indexes_and_values(
    const torch::Tensor& indexes,
    const torch::Tensor& values)
{
    return torch::stack({indexes.to(torch::kFloat64), values.to(torch::kFloat64)}, 1).contiguous();
}

[[nodiscard]] std::vector<CorrelationPoiResult> per_unit_results(
    const CorrelationPayload& correlation,
    const std::vector<std::int64_t>& top_k_values,
    const std::vector<RankBy>& rank_modes)
{
    const auto& grouped = correlation.grouped_correlation();
    const auto& unit_indexes = correlation.unit_indexes();
    const auto channel_count = correlation.channel_count();
    const auto feature_count = correlation.feature_count();

    std::vector<CorrelationPoiResult> results;
    results.reserve(unit_indexes.size());

    for (auto unit_offset = std::int64_t{0}; unit_offset < static_cast<std::int64_t>(unit_indexes.size());
         ++unit_offset) {
        const auto top_k = top_k_values.at(static_cast<std::size_t>(unit_offset));
        validate_top_k(top_k, feature_count);

        std::vector<torch::Tensor> channel_results;
        channel_results.reserve(static_cast<std::size_t>(channel_count));
        for (auto channel_index = std::int64_t{0}; channel_index < channel_count; ++channel_index) {
            const auto correlations = grouped[unit_offset][channel_index];
            const auto scores = contribution_for(correlations, rank_modes.at(static_cast<std::size_t>(channel_index)));
            const auto [selected_scores, selected_indexes] = torch::topk(scores, top_k);
            (void)selected_scores;
            const auto selected_correlations = correlations.index_select(0, selected_indexes);
            channel_results.push_back(pairs_from_indexes_and_values(selected_indexes, selected_correlations));
        }

        results.push_back(CorrelationPoiResult{
            .unit_index = unit_indexes.at(static_cast<std::size_t>(unit_offset)),
            .result = torch::stack(channel_results, 0).contiguous(),
        });
    }

    return results;
}

void copy_target_semantic_metadata(const Buffer& source, Buffer& sink)
{
    static constexpr std::string_view keys[] = {
        "payload.leakage.model",
        "payload.leakage.channels",
        "payload.crypto.algorithm",
        "payload.crypto.state_bytes",
        "payload.trace.count",
        "payload.trace.input",
        "attack.unit.kind",
        "attack.unit.indexes",
        "attack.unit.count",
    };
    for (const auto key : keys) {
        if (source.has_metadata(key)) {
            sink.set_metadata(std::string(key), source.metadata(key));
        }
    }
}

[[nodiscard]] std::string unit_indexes_metadata(const std::vector<std::int64_t>& unit_indexes)
{
    auto value = std::string("[");
    for (std::size_t index = 0; index < unit_indexes.size(); ++index) {
        if (index != 0) {
            value += ",";
        }
        value += std::to_string(unit_indexes[index]);
    }
    value += "]";
    return value;
}

} // namespace

ElementDescriptor PoiSelect::descriptor()
{
    return {
        .type_name = "PoiSelect",
        .klass = "Analyze/SCA/PoI/Select",
        .purpose = "select top-k points of interest per (unit, channel) from a correlation buffer",
        .input_pads = {
            Pad("correlation", PadDirection::Input, Caps(correlation_caps_type)),
        },
        .output_pads = {
            Pad("poi", PadDirection::Output, Caps(correlation_poi_caps_type)),
        },
        .property_specs = {
            PropertySpec("top_k", IntList{10}, "PoI count list; one value applies to all units",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"poi"},
                }),
            PropertySpec("rank_by", StringList{"abs"}, "Ranking mode list per unit: abs, positive, or negative",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"poi"},
                }),
        },
        .keywords = {"poi", "select", "top-k", "correlation", "sca", "statistics"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "routing.element",
                std::string(),
                "element instance name that produced the PoI buffer",
                {"poi"}),
            make_element_metadata_descriptor(
                "payload.poi.method",
                std::string(),
                "PoI selection method",
                {pearson_poi_method_id}),
            make_element_metadata_descriptor(
                "payload.poi.features_count",
                std::int64_t{},
                "number of input features searched for PoI selection",
                {"5000"}),
            make_element_metadata_descriptor(
                "payload.poi.correlation_mode",
                std::string(),
                "effective correlation mode used for the upstream correlation",
                {"recompute", "incremental"}),
            make_element_metadata_descriptor(
                "payload.poi.observation_count",
                std::int64_t{},
                "number of trace observations represented by this PoI result",
                {"1", "50", "10000"}),
            make_element_metadata_descriptor(
                "payload.poi.unit_count", std::int64_t{},
                "number of attack units represented by this PoI result", {"16"}),
            make_element_metadata_descriptor(
                "attack.unit.count", std::int64_t{}, "number of attack units represented", {"16"}),
            make_element_metadata_descriptor(
                "payload.layout", std::string(), "semantic payload layout",
                {"unit/channel/poi/[sample_index,correlation]"}),
        },
    };
}

PoiSelect::PoiSelect(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> PoiSelect::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("PoiSelect requires a connected correlation input");
    }
    if (input->caps().type() != correlation_caps_type) {
        throw std::invalid_argument("PoiSelect requires leakflow/correlation input caps");
    }
    const auto payload = input->payload_as<CorrelationPayload>();
    if (!payload) {
        throw std::invalid_argument("PoiSelect requires a CorrelationPayload");
    }

    const auto unit_count = payload->unit_count();
    const auto rank_modes = rank_modes_for(*this, payload->channel_count());
    auto results = per_unit_results(*payload, per_unit_top_k_for(*this, unit_count), rank_modes);

    Buffer output{Caps(correlation_poi_caps_type)};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "correlation", name());
    copy_target_semantic_metadata(*input, output);
    output.set_metadata("routing.element", name());
    output.set_metadata("payload.poi.method", pearson_poi_method_id);
    output.set_metadata("payload.poi.features_count", std::to_string(payload->feature_count()));
    if (input->has_metadata("payload.correlation.mode")) {
        output.set_metadata("payload.poi.correlation_mode", input->metadata("payload.correlation.mode"));
    }
    output.set_metadata("payload.poi.observation_count", std::to_string(payload->observation_count()));
    output.set_metadata("payload.poi.unit_count", std::to_string(unit_count));
    output.set_metadata("attack.unit.count", std::to_string(unit_count));
    output.set_metadata("attack.unit.indexes", unit_indexes_metadata(payload->unit_indexes()));
    output.set_payload(std::make_shared<CorrelationPoiPayload>(std::move(results), payload->score_name()));

    auto record = make_log_record(log::LogLevel::Debug, "element", "selected Pearson correlation PoIs");
    record.fields.emplace("payload.poi.method", pearson_poi_method_id);
    record.fields.emplace("units", std::to_string(unit_count));
    record.fields.emplace("channel_count", std::to_string(payload->channel_count()));
    record.fields.emplace("features_count", std::to_string(payload->feature_count()));
    record.fields.emplace("observation_count", std::to_string(payload->observation_count()));
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::crypto
