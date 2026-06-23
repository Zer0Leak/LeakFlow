#include "leakflow/plugins/crypto/pearson_poi_finder.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/statistics.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
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

enum class RankBy {
    Abs,
    Positive,
    Negative,
};

struct CorrelationTargetLayout {
    torch::Tensor grouped_correlation;
    std::vector<std::uint16_t> byte_indexes;
    std::int64_t channel_count = 1;
    std::int64_t flattened_target_count = 0;
    std::int64_t feature_count = 0;
};

[[nodiscard]] Caps torch_tensor_caps(Caps::Params params = {})
{
    return Caps(leakflow::base::torch_tensor_caps_type, std::move(params));
}

[[nodiscard]] std::string lower_string(std::string_view text)
{
    std::string lowered;
    lowered.reserve(text.size());
    for (const auto character : text) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    return lowered;
}

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

[[nodiscard]] double double_property_or(const Element& element, std::string_view name, double fallback)
{
    if (const auto value = element.property_as<double>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] leakflow::base::PearsonComputeDtype compute_dtype_for(std::string_view text)
{
    const auto lowered = lower_string(text);
    if (lowered == "input") {
        return leakflow::base::PearsonComputeDtype::Input;
    }
    if (lowered == "float32") {
        return leakflow::base::PearsonComputeDtype::Float32;
    }
    if (lowered == "float64") {
        return leakflow::base::PearsonComputeDtype::Float64;
    }

    throw std::invalid_argument("PearsonPoiFinder compute_dtype must be input, float32, or float64");
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

    throw std::invalid_argument("PearsonPoiFinder rank_by values must be abs, positive, or negative");
}

// rank_by is applied per leakage feature/channel (the last target dimension, for
// example HW(m), HW(y)). It must be one value (applied to every feature) or one
// value per feature. The returned vector is indexed by channel.
[[nodiscard]] std::vector<RankBy> rank_modes_for(const Element& element, std::int64_t channel_count)
{
    auto values = string_list_property_or(element, "rank_by", StringList{"abs"});
    if (values.empty()) {
        throw std::invalid_argument("PearsonPoiFinder rank_by cannot be empty");
    }

    std::vector<RankBy> modes;
    if (values.size() == 1) {
        modes.assign(static_cast<std::size_t>(channel_count), rank_by_for(values.front()));
        return modes;
    }
    if (values.size() != static_cast<std::size_t>(channel_count)) {
        throw std::invalid_argument(
            "PearsonPoiFinder rank_by must have length 1 or the number of leakage features/channels");
    }

    modes.reserve(values.size());
    for (const auto& value : values) {
        modes.push_back(rank_by_for(value));
    }
    return modes;
}

[[nodiscard]] std::vector<std::int64_t> per_result_group_top_k_for(const Element& element, std::int64_t result_group_count)
{
    auto values = int_list_property_or(element, "top_k", IntList{10});
    if (values.empty()) {
        throw std::invalid_argument("PearsonPoiFinder top_k cannot be empty");
    }

    if (values.size() == 1) {
        return std::vector<std::int64_t>(static_cast<std::size_t>(result_group_count), values.front());
    }
    if (values.size() != static_cast<std::size_t>(result_group_count)) {
        throw std::invalid_argument("PearsonPoiFinder top_k must have length 1 or result group count");
    }

    return values;
}

void validate_top_k(std::int64_t top_k, std::int64_t feature_count)
{
    if (top_k <= 0) {
        throw std::invalid_argument("PearsonPoiFinder top_k values must be positive");
    }
    if (top_k > feature_count) {
        throw std::invalid_argument("PearsonPoiFinder top_k values must not exceed feature count");
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

    throw std::invalid_argument("unsupported PearsonPoiFinder rank_by");
}

[[nodiscard]] torch::Tensor pairs_from_indexes_and_values(
    const torch::Tensor& indexes,
    const torch::Tensor& values)
{
    return torch::stack({indexes.to(torch::kFloat64), values.to(torch::kFloat64)}, 1).contiguous();
}

[[nodiscard]] std::vector<CorrelationPoiResult> per_byte_results(
    const CorrelationTargetLayout& layout,
    const std::vector<std::int64_t>& top_k_values,
    const std::vector<RankBy>& rank_modes)
{
    std::vector<CorrelationPoiResult> results;
    results.reserve(layout.byte_indexes.size());

    for (auto byte_offset = std::int64_t{0};
         byte_offset < static_cast<std::int64_t>(layout.byte_indexes.size());
         ++byte_offset) {
        const auto top_k = top_k_values.at(static_cast<std::size_t>(byte_offset));
        validate_top_k(top_k, layout.feature_count);

        std::vector<torch::Tensor> channel_results;
        channel_results.reserve(static_cast<std::size_t>(layout.channel_count));
        for (auto channel_index = std::int64_t{0}; channel_index < layout.channel_count; ++channel_index) {
            const auto correlations = layout.grouped_correlation[byte_offset][channel_index];
            const auto scores = contribution_for(correlations, rank_modes.at(static_cast<std::size_t>(channel_index)));
            const auto [selected_scores, selected_indexes] = torch::topk(scores, top_k);
            (void)selected_scores;
            const auto selected_correlations = correlations.index_select(0, selected_indexes);
            channel_results.push_back(pairs_from_indexes_and_values(selected_indexes, selected_correlations));
        }

        results.push_back(CorrelationPoiResult{
            .target_byte_index = layout.byte_indexes.at(static_cast<std::size_t>(byte_offset)),
            .result = torch::stack(channel_results, 0).contiguous(),
        });
    }

    return results;
}

void copy_target_semantic_metadata(const Buffer& source, Buffer& sink)
{
    static constexpr std::string_view keys[] = {
        "payload.leakage.model",
        "payload.leakage.byte_indexes",
        "payload.leakage.channels",
        "payload.crypto.algorithm",
        "payload.crypto.state_bytes",
        "payload.trace.count",
        "payload.trace.input",
    };

    for (const auto key : keys) {
        if (source.has_metadata(key)) {
            sink.set_metadata(std::string(key), source.metadata(key));
        }
    }
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

[[nodiscard]] std::vector<std::int64_t> parse_int_metadata_list(std::string_view text)
{
    auto trimmed = trim_to_string(text);
    if (trimmed.size() >= 2 && trimmed.front() == '[' && trimmed.back() == ']') {
        trimmed = trimmed.substr(1, trimmed.size() - 2);
    }
    if (trimmed.empty()) {
        return {};
    }

    std::vector<std::int64_t> values;
    for (const auto& part : split_comma_list(trimmed)) {
        std::size_t consumed = 0;
        const auto value = std::stoll(part, &consumed);
        if (consumed != part.size()) {
            throw std::invalid_argument("invalid integer metadata list value");
        }
        values.push_back(value);
    }
    return values;
}

[[nodiscard]] std::vector<std::string> target_channels_from_metadata(const Buffer& targets_buffer)
{
    if (!targets_buffer.has_metadata("payload.leakage.channels")) {
        return {};
    }

    return split_comma_list(targets_buffer.metadata("payload.leakage.channels"));
}

[[nodiscard]] std::vector<std::int64_t> byte_indexes_from_metadata(const Buffer& targets_buffer)
{
    if (!targets_buffer.has_metadata("payload.leakage.byte_indexes")) {
        return {};
    }

    return parse_int_metadata_list(targets_buffer.metadata("payload.leakage.byte_indexes"));
}

[[nodiscard]] std::uint16_t checked_byte_index(std::int64_t value)
{
    if (value < 0 || value > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument("PearsonPoiFinder byte indexes must fit in uint16");
    }

    return static_cast<std::uint16_t>(value);
}

[[nodiscard]] std::vector<std::uint16_t> checked_byte_indexes(std::vector<std::int64_t> values)
{
    std::vector<std::uint16_t> indexes;
    indexes.reserve(values.size());
    for (const auto value : values) {
        indexes.push_back(checked_byte_index(value));
    }
    return indexes;
}

[[nodiscard]] std::vector<std::uint16_t> default_byte_indexes(std::int64_t count)
{
    std::vector<std::uint16_t> indexes;
    indexes.reserve(static_cast<std::size_t>(count));
    for (auto index = std::int64_t{0}; index < count; ++index) {
        indexes.push_back(checked_byte_index(index));
    }
    return indexes;
}

[[nodiscard]] CorrelationTargetLayout correlation_target_layout(
    const torch::Tensor& correlation,
    const Buffer& targets_buffer)
{
    const auto feature_count = correlation.size(correlation.dim() - 1);
    auto flattened_correlation = correlation.reshape({-1, feature_count});
    const auto flattened_target_count = flattened_correlation.size(0);

    auto byte_indexes = checked_byte_indexes(byte_indexes_from_metadata(targets_buffer));
    const auto channels = target_channels_from_metadata(targets_buffer);
    auto channel_count = std::int64_t{1};

    if (!byte_indexes.empty()) {
        channel_count = channels.empty() ? std::int64_t{1} : static_cast<std::int64_t>(channels.size());
        const auto represented_target_count =
            static_cast<std::int64_t>(byte_indexes.size()) * channel_count;
        if (represented_target_count != flattened_target_count) {
            throw std::invalid_argument(
                "PearsonPoiFinder leakage byte indexes and channels must match flattened target count");
        }
    } else {
        byte_indexes = default_byte_indexes(flattened_target_count);
    }

    return CorrelationTargetLayout{
        .grouped_correlation =
            flattened_correlation
                .reshape({static_cast<std::int64_t>(byte_indexes.size()), channel_count, feature_count})
                .contiguous(),
        .byte_indexes = std::move(byte_indexes),
        .channel_count = channel_count,
        .flattened_target_count = flattened_target_count,
        .feature_count = feature_count,
    };
}

} // namespace

ElementDescriptor PearsonPoiFinder::descriptor()
{
    return {
        .type_name = "PearsonPoiFinder",
        .klass = "Analyze/SCA/Statistics/PoI",
        .purpose = "select points of interest from Pearson correlations between features and target models",
        .input_pads = {
            Pad(
                "features",
                PadDirection::Input,
                torch_tensor_caps({{leakflow::base::caps_param_rank, "2"}})),
            Pad("targets", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .output_pads = {
            Pad("poi", PadDirection::Output, Caps(correlation_poi_caps_type)),
        },
        .property_specs = {
            PropertySpec("top_k", IntList{10}, "PoI count list; one value applies to all result groups",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"poi"},
                }),
            PropertySpec("rank_by", StringList{"abs"}, "Ranking mode list per result group: abs, positive, or negative",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"poi"},
                }),
            PropertySpec(
                "compute_dtype",
                std::string("input"),
                "Pearson compute dtype",
                "",
                StringEnumConstraint{{"input", "float32", "float64"}},
                "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"poi"},
                }),
            PropertySpec(
                "epsilon",
                1.0e-12,
                "Small positive denominator guard for Pearson correlation",
                "",
                std::monostate{},
                "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"poi"},
                }),
        },
        .keywords = {"pearson", "correlation", "poi", "sca", "statistics"},
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
        },
        .metadata_suggestions = {
            make_element_metadata_descriptor(
                "payload.leakage.byte_indexes",
                IntList{},
                "byte indexes represented by custom target tensors",
                {"[3,5]"}),
            make_element_metadata_descriptor(
                "payload.leakage.channels",
                StringList{},
                "target channel names represented by custom target tensors",
                {"HW(m),HW(y)"}),
            make_element_metadata_descriptor(
                "payload.crypto.algorithm",
                std::string(),
                "algorithm name carried into PoI results",
                {"AES"}),
        },
    };
}

PearsonPoiFinder::PearsonPoiFinder(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> PearsonPoiFinder::process(std::optional<Buffer>)
{
    throw std::invalid_argument("PearsonPoiFinder requires named features and targets inputs");
}

std::optional<Buffer> PearsonPoiFinder::process_inputs(ElementInputs inputs)
{
    const auto& features_buffer = required_input(inputs, "features", "PearsonPoiFinder");
    const auto& targets_buffer = required_input(inputs, "targets", "PearsonPoiFinder");
    const auto features_payload = torch_payload_for(features_buffer, "features");
    const auto targets_payload = torch_payload_for(targets_buffer, "targets");

    leakflow::base::PearsonCorrelationOptions correlation_options;
    correlation_options.compute_dtype = compute_dtype_for(string_property_or(*this, "compute_dtype", "input"));
    correlation_options.epsilon = double_property_or(*this, "epsilon", 1.0e-12);

    const auto correlation = leakflow::base::pearson_correlation(
        features_payload->tensor(),
        targets_payload->tensor(),
        correlation_options);
    const auto layout = correlation_target_layout(correlation, targets_buffer);
    const auto result_group_count = static_cast<std::int64_t>(layout.byte_indexes.size());
    const auto rank_modes = rank_modes_for(*this, layout.channel_count);

    auto results =
        per_byte_results(layout, per_result_group_top_k_for(*this, result_group_count), rank_modes);
    const auto score_name = std::string("correlation");

    Buffer output{Caps(correlation_poi_caps_type)};
    forward_metadata(inputs, profile_for_klass(element_kclass()), output);
    copy_target_semantic_metadata(targets_buffer, output);
    output.set_metadata("routing.element", name());
    output.set_metadata("payload.poi.method", pearson_poi_method_id);
    output.set_metadata("payload.poi.features_count", std::to_string(layout.feature_count));
    output.set_payload(std::make_shared<CorrelationPoiPayload>(std::move(results), score_name));

    auto record = make_log_record(log::LogLevel::Debug, "element", "selected Pearson correlation PoIs");
    record.fields.emplace("payload.poi.method", pearson_poi_method_id);
    record.fields.emplace("score", score_name);
    record.fields.emplace("result_groups", std::to_string(result_group_count));
    record.fields.emplace("channel_count", std::to_string(layout.channel_count));
    record.fields.emplace("target_count", std::to_string(layout.flattened_target_count));
    record.fields.emplace("features_count", std::to_string(layout.feature_count));
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::crypto
