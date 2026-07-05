#include "leakflow/plugins/crypto/pearson_correlator.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/statistics.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/correlation_payload.hpp"

#include <cctype>
#include <cstdint>
#include <limits>
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

enum class CorrelationMode {
    Auto,
    Recompute,
    Incremental,
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
    throw std::invalid_argument("PearsonCorrelator compute_dtype must be input, float32, or float64");
}

[[nodiscard]] CorrelationMode correlation_mode_for(std::string_view text)
{
    const auto lowered = lower_string(text);
    if (lowered == "auto") {
        return CorrelationMode::Auto;
    }
    if (lowered == "recompute") {
        return CorrelationMode::Recompute;
    }
    if (lowered == "incremental") {
        return CorrelationMode::Incremental;
    }
    throw std::invalid_argument("PearsonCorrelator correlation_mode must be auto, recompute, or incremental");
}

[[nodiscard]] CorrelationMode effective_correlation_mode(CorrelationMode configured_mode, bool live_driven)
{
    if (configured_mode == CorrelationMode::Auto) {
        return live_driven ? CorrelationMode::Incremental : CorrelationMode::Recompute;
    }
    return configured_mode;
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
        throw std::invalid_argument("PearsonCorrelator byte indexes must fit in uint16");
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
                "PearsonCorrelator leakage byte indexes and channels must match flattened target count");
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

ElementDescriptor PearsonCorrelator::descriptor()
{
    return {
        .type_name = "PearsonCorrelator",
        .klass = "Analyze/SCA/Statistics/Correlation",
        .purpose = "compute the Pearson correlation of features against target models (recompute or incremental)",
        .input_pads = {
            Pad(
                "features",
                PadDirection::Input,
                torch_tensor_caps({{leakflow::base::caps_param_rank, "2"}})),
            Pad("targets", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .output_pads = {
            Pad("correlation", PadDirection::Output, Caps(correlation_caps_type)),
        },
        .property_specs = {
            PropertySpec(
                "correlation_mode",
                std::string("auto"),
                "Correlation mode: auto follows upstream liveness; recompute uses only the current buffer; "
                "incremental accumulates across buffers",
                "",
                StringEnumConstraint{{"auto", "recompute", "incremental"}},
                "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"correlation"},
                }),
            PropertySpec(
                "active_correlation_mode",
                std::string("recompute"),
                "Resolved correlation mode currently selected by correlation_mode",
                "",
                StringEnumConstraint{{"recompute", "incremental"}},
                "",
                PropertyEffect{},
                false,
                false),
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
                    .output_pads = {"correlation"},
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
                    .output_pads = {"correlation"},
                }),
        },
        .keywords = {"pearson", "correlation", "sca", "statistics"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "routing.element",
                std::string(),
                "element instance name that produced the correlation buffer",
                {"correlator"}),
            make_element_metadata_descriptor(
                "payload.correlation.method",
                std::string(),
                "correlation method",
                {pearson_correlation_method_id}),
            make_element_metadata_descriptor(
                "payload.correlation.features_count",
                std::int64_t{},
                "number of input features correlated",
                {"5000"}),
            make_element_metadata_descriptor(
                "payload.correlation.mode",
                std::string(),
                "effective correlation mode used for this buffer",
                {"recompute", "incremental"}),
            make_element_metadata_descriptor(
                "payload.correlation.observation_count",
                std::int64_t{},
                "number of trace observations represented by this correlation",
                {"1", "50", "10000"}),
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
                "algorithm name carried into correlation results",
                {"AES"}),
        },
    };
}

PearsonCorrelator::PearsonCorrelator(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

void PearsonCorrelator::start()
{
    update_active_correlation_mode();
    reset_incremental_correlation();
}

std::optional<Buffer> PearsonCorrelator::process(std::optional<Buffer>)
{
    throw std::invalid_argument("PearsonCorrelator requires named features and targets inputs");
}

std::optional<Buffer> PearsonCorrelator::process_inputs(ElementInputs inputs)
{
    const auto& features_buffer = required_input(inputs, "features", "PearsonCorrelator");
    const auto& targets_buffer = required_input(inputs, "targets", "PearsonCorrelator");
    const auto features_payload = torch_payload_for(features_buffer, "features");
    const auto targets_payload = torch_payload_for(targets_buffer, "targets");

    leakflow::base::PearsonCorrelationOptions correlation_options;
    correlation_options.compute_dtype = compute_dtype_for(string_property_or(*this, "compute_dtype", "input"));
    correlation_options.epsilon = double_property_or(*this, "epsilon", 1.0e-12);

    update_active_correlation_mode();
    const auto correlation_mode = correlation_mode_for(
        string_property_or(*this, "active_correlation_mode", "recompute"));
    torch::Tensor correlation;
    std::int64_t observation_count = features_payload->tensor().size(0);

    if (correlation_mode == CorrelationMode::Recompute) {
        reset_incremental_correlation();
        correlation = leakflow::base::pearson_correlation(
            features_payload->tensor(),
            targets_payload->tensor(),
            correlation_options);
    } else {
        const auto options_changed =
            !incremental_options_
            || incremental_options_->compute_dtype != correlation_options.compute_dtype
            || incremental_options_->epsilon != correlation_options.epsilon;
        if (!incremental_correlation_ || options_changed) {
            incremental_correlation_.emplace(correlation_options);
            incremental_options_ = correlation_options;
        }
        correlation = incremental_correlation_->update(
            features_payload->tensor(),
            targets_payload->tensor());
        observation_count = incremental_correlation_->observation_count();
    }

    const auto layout = correlation_target_layout(correlation, targets_buffer);
    const auto score_name = std::string("correlation");

    Buffer output{Caps(correlation_caps_type)};
    forward_metadata(inputs, profile_for_klass(element_kclass()), output, name());
    copy_target_semantic_metadata(targets_buffer, output);
    output.set_metadata("routing.element", name());
    output.set_metadata("payload.correlation.method", pearson_correlation_method_id);
    output.set_metadata("payload.correlation.features_count", std::to_string(layout.feature_count));
    output.set_metadata(
        "payload.correlation.mode",
        correlation_mode == CorrelationMode::Recompute ? "recompute" : "incremental");
    output.set_metadata("payload.correlation.observation_count", std::to_string(observation_count));
    output.set_payload(std::make_shared<CorrelationPayload>(
        layout.grouped_correlation,
        layout.byte_indexes,
        layout.channel_count,
        layout.feature_count,
        score_name,
        observation_count));

    auto record = make_log_record(log::LogLevel::Debug, "element", "computed Pearson correlation");
    record.fields.emplace("payload.correlation.method", pearson_correlation_method_id);
    record.fields.emplace("byte_groups", std::to_string(layout.byte_indexes.size()));
    record.fields.emplace("channel_count", std::to_string(layout.channel_count));
    record.fields.emplace("features_count", std::to_string(layout.feature_count));
    record.fields.emplace("correlation_mode", output.metadata("payload.correlation.mode"));
    record.fields.emplace("observation_count", std::to_string(observation_count));
    leakflow::log::write(std::move(record));

    return output;
}

bool PearsonCorrelator::can_replay() const
{
    return string_property_or(*this, "active_correlation_mode", "recompute") == "recompute";
}

void PearsonCorrelator::reset_incremental_correlation()
{
    incremental_correlation_.reset();
    incremental_options_.reset();
}

void PearsonCorrelator::update_active_correlation_mode()
{
    const auto configured_mode =
        correlation_mode_for(string_property_or(*this, "correlation_mode", "auto"));
    const auto active_mode = effective_correlation_mode(configured_mode, is_live_driven());
    set_read_only_property(
        "active_correlation_mode",
        std::string(active_mode == CorrelationMode::Recompute ? "recompute" : "incremental"));
}

void PearsonCorrelator::property_changed(std::string_view name)
{
    if (name == "correlation_mode") {
        update_active_correlation_mode();
    }
}

void PearsonCorrelator::live_driven_changed()
{
    update_active_correlation_mode();
}

} // namespace leakflow::plugins::crypto
