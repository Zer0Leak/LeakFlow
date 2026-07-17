#include "leakflow/plugins/crypto/poi_select.hpp"

#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/correlation_payload.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include <algorithm>
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

// `channel_columns` selects which correlation channel columns to keep, in output order;
// `rank_modes` is still indexed by the original correlation column (a channel's ranking mode
// is a property of the leakage channel, not of its output position).
[[nodiscard]] std::vector<CorrelationPoiResult> per_unit_results(
    const CorrelationPayload& correlation,
    const std::vector<std::int64_t>& top_k_values,
    const std::vector<RankBy>& rank_modes,
    const std::vector<std::int64_t>& channel_columns)
{
    const auto& grouped = correlation.grouped_correlation();
    const auto& units = correlation.units();
    const auto feature_count = correlation.feature_count();

    std::vector<CorrelationPoiResult> results;
    results.reserve(units.size());

    for (auto unit_offset = std::int64_t{0}; unit_offset < static_cast<std::int64_t>(units.size());
         ++unit_offset) {
        const auto top_k = top_k_values.at(static_cast<std::size_t>(unit_offset));
        validate_top_k(top_k, feature_count);

        std::vector<torch::Tensor> channel_results;
        channel_results.reserve(channel_columns.size());
        for (const auto column : channel_columns) {
            const auto correlations = grouped[unit_offset][column];
            const auto scores = contribution_for(correlations, rank_modes.at(static_cast<std::size_t>(column)));
            const auto [selected_scores, selected_indexes] = torch::topk(scores, top_k);
            (void)selected_scores;
            const auto selected_correlations = correlations.index_select(0, selected_indexes);
            channel_results.push_back(pairs_from_indexes_and_values(selected_indexes, selected_correlations));
        }

        results.push_back(CorrelationPoiResult{
            .unit_index = units.at(static_cast<std::size_t>(unit_offset)),
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

// Keep only the requested unit ids, in requested order; error on a missing one. An
// empty request keeps every unit in payload order.
[[nodiscard]] std::vector<CorrelationPoiResult> select_units(
    std::vector<CorrelationPoiResult> results, const std::vector<std::int64_t>& requested)
{
    if (requested.empty()) {
        return results;
    }
    std::vector<CorrelationPoiResult> selected;
    selected.reserve(requested.size());
    for (const auto unit : requested) {
        const auto found = std::find_if(results.begin(), results.end(),
            [unit](const CorrelationPoiResult& result) { return result.unit_index == unit; });
        if (found == results.end()) {
            throw std::invalid_argument(
                "PoiSelect units: requested unit " + std::to_string(unit) + " is not in the correlation");
        }
        selected.push_back(*found);
    }
    return selected;
}

[[nodiscard]] std::string units_metadata(const std::vector<std::int64_t>& units)
{
    auto value = std::string("[");
    for (std::size_t index = 0; index < units.size(); ++index) {
        if (index != 0) {
            value += ",";
        }
        value += std::to_string(units[index]);
    }
    value += "]";
    return value;
}

// Bare (bracket-less) channel projection for the payload.leakage.channels metadata, matching
// the convention AesLeakage writes and PoiSelect parses back.
[[nodiscard]] std::string channels_metadata(const Channels& channels)
{
    std::string value;
    for (auto index = std::int64_t{0}; index < channels.size(); ++index) {
        if (index != 0) {
            value += ",";
        }
        value += channels.at(index);
    }
    return value;
}

struct ChannelSelection {
    std::vector<std::int64_t> columns; // correlation column per output channel, in output order
    Channels channels;                 // labels in output order (none when unlabeled and kept-all)
};

// Resolve which correlation channel columns to keep. An empty `channels` property keeps every
// column in correlation order (carrying the identity through unchanged, possibly none). A
// non-empty one keeps/reorders channels by name and therefore requires a named channel identity;
// a requested name absent from the correlation is an error.
[[nodiscard]] ChannelSelection channel_selection_for(
    const Element& element, const Channels& identity, std::int64_t channel_count)
{
    const auto requested = string_list_property_or(element, "channels", StringList{});
    ChannelSelection selection;
    if (requested.empty()) {
        selection.columns.reserve(static_cast<std::size_t>(channel_count));
        for (auto column = std::int64_t{0}; column < channel_count; ++column) {
            selection.columns.push_back(column);
        }
        selection.channels = identity;
        return selection;
    }
    if (identity.empty()) {
        throw std::invalid_argument(
            "PoiSelect channels: the correlation carries no channel identity to select by name");
    }
    if (identity.size() != channel_count) {
        throw std::invalid_argument(
            "PoiSelect channels: correlation channel identity does not match the channel count");
    }
    selection.columns.reserve(requested.size());
    std::vector<std::string> labels;
    labels.reserve(requested.size());
    for (const auto& name : requested) {
        auto column = std::int64_t{-1};
        for (auto candidate = std::int64_t{0}; candidate < identity.size(); ++candidate) {
            if (identity.at(candidate) == name) {
                column = candidate;
                break;
            }
        }
        if (column < 0) {
            throw std::invalid_argument(
                "PoiSelect channels: requested channel " + name + " is not in the correlation");
        }
        selection.columns.push_back(column);
        labels.push_back(name);
    }
    selection.channels = Channels::of(std::move(labels));
    return selection;
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
            PropertySpec("units", Units::none(),
                "unit indexes to keep, e.g. [0] / [0:16] / [0,2:4]; none/[] = all, in correlation order",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"poi"},
                }),
            PropertySpec("channels", StringList{},
                "leakage channels to keep/reorder by name, e.g. [HW(y)] or [HW(m),HW(y)]; "
                "[] = all, in correlation order",
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

    // The correlation's channel identity comes from the typed axis, falling back to its
    // payload.leakage.channels metadata projection (e.g. on a reloaded buffer).
    Channels channel_identity = input->channels();
    if (channel_identity.empty() && input->has_metadata("payload.leakage.channels")) {
        channel_identity = Channels::parse(input->metadata("payload.leakage.channels"));
    }
    const auto channel_selection = channel_selection_for(*this, channel_identity, payload->channel_count());

    auto results = per_unit_results(
        *payload, per_unit_top_k_for(*this, unit_count), rank_modes, channel_selection.columns);
    const auto requested_units = property_as<Units>("units").value_or(Units::none()).to_vector();
    results = select_units(std::move(results), requested_units);

    std::vector<std::int64_t> selected_units;
    selected_units.reserve(results.size());
    for (const auto& result : results) {
        selected_units.push_back(result.unit_index);
    }

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
    output.set_metadata("payload.poi.unit_count", std::to_string(selected_units.size()));
    output.set_metadata("attack.unit.count", std::to_string(selected_units.size()));
    output.set_metadata("attack.unit.indexes", units_metadata(selected_units));
    // Carry the semantic axes as first-class typed values: the selected units and the
    // selected channels. When a channel subset/reorder was requested, re-project the
    // authoritative axis into the payload.leakage.channels metadata so the two agree
    // (copy_target_semantic_metadata forwarded the correlation's original list above).
    output.set_units(Units::of(selected_units));
    output.set_channels(channel_selection.channels);
    if (!channel_selection.channels.empty()) {
        output.set_metadata("payload.leakage.channels", channels_metadata(channel_selection.channels));
    }
    output.set_payload(std::make_shared<CorrelationPoiPayload>(std::move(results), payload->score_name()));

    auto record = make_log_record(log::LogLevel::Debug, "element", "selected Pearson correlation PoIs");
    record.fields.emplace("payload.poi.method", pearson_poi_method_id);
    record.fields.emplace("units", std::to_string(selected_units.size()));
    record.fields.emplace("channel_count", std::to_string(payload->channel_count()));
    record.fields.emplace("features_count", std::to_string(payload->feature_count()));
    record.fields.emplace("observation_count", std::to_string(payload->observation_count()));
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::crypto
