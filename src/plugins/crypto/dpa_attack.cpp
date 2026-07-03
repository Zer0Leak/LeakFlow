#include "leakflow/plugins/crypto/dpa_attack.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/attack_payload.hpp"

#include <c10/core/ScalarType.h>

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

enum class ScoreMethod {
    MaxAbs,
    MaxPositive,
    MaxNegative,
};

enum class ScoreChannels {
    GuessDependent,
    All,
};

enum class AccumulationMode {
    Auto,
    Recompute,
    Incremental,
};

enum class ComputeDtype {
    Input,
    Float32,
    Float64,
};

struct PreparedInputs {
    torch::Tensor features;
    torch::Tensor hypotheses;
    std::int64_t unit_count = 0;
    std::int64_t guess_count = 0;
    std::int64_t trace_count = 0;
    std::int64_t channel_count = 0;
    std::int64_t sample_count = 0;
    torch::ScalarType dtype = torch::kFloat32;
};

struct DifferenceSums {
    std::int64_t count = 0;
    torch::Tensor sum_x;
    torch::Tensor group1_count;
    torch::Tensor group1_sum;
};

struct ScoreResult {
    torch::Tensor scores;
    torch::Tensor ranking;
    torch::Tensor best_guess_index;
    torch::Tensor best_score;
    torch::Tensor best_channel;
    torch::Tensor best_sample;
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

[[nodiscard]] std::vector<std::int64_t> parse_int_list(std::string_view text)
{
    const auto trimmed = trim_to_string(text);
    if (trimmed.empty() || trimmed.front() != '[' || trimmed.back() != ']') {
        throw std::invalid_argument("DpaAttack metadata integer lists must use [..] syntax");
    }

    const auto body = std::string_view(trimmed).substr(1, trimmed.size() - 2);
    if (trim_to_string(body).empty()) {
        return {};
    }

    std::vector<std::int64_t> values;
    for (const auto& part : split_comma_list(body)) {
        std::size_t parsed = 0;
        const auto value = std::stoll(part, &parsed, 10);
        if (parsed != part.size()) {
            throw std::invalid_argument("DpaAttack metadata integer list contained a non-integer value");
        }
        values.push_back(value);
    }
    return values;
}

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

[[nodiscard]] ScoreMethod score_method_for(std::string_view text)
{
    const auto lowered = lower_string(text);
    if (lowered == "max_abs") {
        return ScoreMethod::MaxAbs;
    }
    if (lowered == "max_positive") {
        return ScoreMethod::MaxPositive;
    }
    if (lowered == "max_negative") {
        return ScoreMethod::MaxNegative;
    }
    throw std::invalid_argument("DpaAttack score_method must be max_abs, max_positive, or max_negative");
}

[[nodiscard]] ScoreChannels score_channels_for(std::string_view text)
{
    const auto lowered = lower_string(text);
    if (lowered == "guess_dependent") {
        return ScoreChannels::GuessDependent;
    }
    if (lowered == "all") {
        return ScoreChannels::All;
    }
    throw std::invalid_argument("DpaAttack score_channels must be guess_dependent or all");
}

[[nodiscard]] AccumulationMode accumulation_mode_for(std::string_view text)
{
    const auto lowered = lower_string(text);
    if (lowered == "auto") {
        return AccumulationMode::Auto;
    }
    if (lowered == "recompute") {
        return AccumulationMode::Recompute;
    }
    if (lowered == "incremental") {
        return AccumulationMode::Incremental;
    }
    throw std::invalid_argument("DpaAttack accumulation_mode must be auto, recompute, or incremental");
}

[[nodiscard]] AccumulationMode effective_accumulation_mode(AccumulationMode configured_mode, bool live_driven)
{
    if (configured_mode == AccumulationMode::Auto) {
        return live_driven ? AccumulationMode::Incremental : AccumulationMode::Recompute;
    }
    return configured_mode;
}

[[nodiscard]] ComputeDtype compute_dtype_for(std::string_view text)
{
    const auto lowered = lower_string(text);
    if (lowered == "input") {
        return ComputeDtype::Input;
    }
    if (lowered == "float32") {
        return ComputeDtype::Float32;
    }
    if (lowered == "float64") {
        return ComputeDtype::Float64;
    }
    throw std::invalid_argument("DpaAttack compute_dtype must be input, float32, or float64");
}

[[nodiscard]] torch::ScalarType tensor_compute_dtype(torch::ScalarType input_dtype, ComputeDtype compute_dtype)
{
    switch (compute_dtype) {
    case ComputeDtype::Input:
        if (!c10::isFloatingType(input_dtype)) {
            throw std::invalid_argument("DpaAttack input compute_dtype requires floating-point features");
        }
        return input_dtype;
    case ComputeDtype::Float32:
        return torch::kFloat32;
    case ComputeDtype::Float64:
        return torch::kFloat64;
    }
    throw std::invalid_argument("unsupported DpaAttack compute dtype");
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

void require_strided_tensor(const torch::Tensor& tensor, std::string_view name)
{
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " tensor must be defined");
    }
    if (tensor.layout() != torch::kStrided) {
        throw std::invalid_argument(std::string(name) + " tensor must use strided layout");
    }
}

void require_numeric_tensor(const torch::Tensor& tensor, std::string_view name)
{
    require_strided_tensor(tensor, name);
    if (!c10::isFloatingType(tensor.scalar_type()) && !c10::isIntegralType(tensor.scalar_type(), true)) {
        throw std::invalid_argument(std::string(name) + " tensor must be numeric");
    }
}

void require_binary_hypotheses(const torch::Tensor& tensor)
{
    const auto is_zero = tensor == 0;
    const auto is_one = tensor == 1;
    if (!torch::logical_or(is_zero, is_one).all().item<bool>()) {
        throw std::invalid_argument("DpaAttack hypotheses tensor must be binary 0/1");
    }
}

[[nodiscard]] PreparedInputs prepare_inputs(
    const torch::Tensor& feature_tensor,
    const torch::Tensor& hypothesis_tensor,
    ComputeDtype compute_dtype)
{
    require_numeric_tensor(feature_tensor, "features");
    require_numeric_tensor(hypothesis_tensor, "hypotheses");

    if (feature_tensor.device() != hypothesis_tensor.device()) {
        throw std::invalid_argument("DpaAttack features and hypotheses must be on the same device");
    }
    if (hypothesis_tensor.dim() != 4) {
        throw std::invalid_argument("DpaAttack hypotheses tensor must have shape [U,G,N,L]");
    }

    const auto unit_count = hypothesis_tensor.size(0);
    const auto guess_count = hypothesis_tensor.size(1);
    const auto trace_count = hypothesis_tensor.size(2);
    const auto channel_count = hypothesis_tensor.size(3);
    if (unit_count <= 0 || guess_count <= 0 || trace_count <= 0 || channel_count <= 0) {
        throw std::invalid_argument("DpaAttack hypotheses tensor must have positive [U,G,N,L] axes");
    }
    require_binary_hypotheses(hypothesis_tensor);

    std::int64_t sample_count = 0;
    torch::Tensor features;
    if (feature_tensor.dim() == 2) {
        if (feature_tensor.size(0) != trace_count) {
            throw std::invalid_argument("DpaAttack features [N,S] must match hypothesis trace count");
        }
        sample_count = feature_tensor.size(1);
        features = feature_tensor.unsqueeze(0).expand({unit_count, trace_count, sample_count});
    } else if (feature_tensor.dim() == 3) {
        if (feature_tensor.size(0) != unit_count || feature_tensor.size(1) != trace_count) {
            throw std::invalid_argument("DpaAttack features [U,N,S] must match hypotheses [U,G,N,L]");
        }
        sample_count = feature_tensor.size(2);
        features = feature_tensor;
    } else {
        throw std::invalid_argument("DpaAttack features tensor must have shape [N,S] or [U,N,S]");
    }
    if (sample_count <= 0) {
        throw std::invalid_argument("DpaAttack features tensor must have at least one sample");
    }

    const auto dtype = tensor_compute_dtype(feature_tensor.scalar_type(), compute_dtype);
    return PreparedInputs{
        .features = features.to(dtype).contiguous(),
        .hypotheses = hypothesis_tensor.to(dtype).contiguous(),
        .unit_count = unit_count,
        .guess_count = guess_count,
        .trace_count = trace_count,
        .channel_count = channel_count,
        .sample_count = sample_count,
        .dtype = dtype,
    };
}

[[nodiscard]] DifferenceSums difference_sums(const PreparedInputs& prepared)
{
    const auto hypothesis_flat = prepared.hypotheses.permute({0, 1, 3, 2})
                                     .reshape({prepared.unit_count, prepared.guess_count * prepared.channel_count,
                                         prepared.trace_count});
    const auto group1_sum = torch::bmm(hypothesis_flat, prepared.features)
                                .reshape({prepared.unit_count, prepared.guess_count, prepared.channel_count,
                                    prepared.sample_count})
                                .contiguous();

    return DifferenceSums{
        .count = prepared.trace_count,
        .sum_x = prepared.features.sum(1).contiguous(),
        .group1_count = prepared.hypotheses.sum(2).contiguous(),
        .group1_sum = group1_sum,
    };
}

[[nodiscard]] torch::Tensor difference_from_sums(const DifferenceSums& sums)
{
    if (sums.count <= 0) {
        throw std::invalid_argument("DpaAttack difference requires at least one observation");
    }

    const auto count1 = sums.group1_count;
    const auto count0 = static_cast<double>(sums.count) - count1;
    const auto valid = torch::logical_and(count1 >= 1, count0 >= 1).unsqueeze(-1);
    const auto count1_safe = count1.clamp_min(1).unsqueeze(-1);
    const auto count0_safe = count0.clamp_min(1).unsqueeze(-1);
    const auto inv0 = torch::reciprocal(count0_safe);
    const auto difference =
        sums.group1_sum * (torch::reciprocal(count1_safe) + inv0)
        - sums.sum_x.unsqueeze(1).unsqueeze(1) * inv0;
    return torch::where(valid, difference, torch::zeros_like(difference)).contiguous();
}

[[nodiscard]] std::vector<std::string> channels_from_metadata(const Buffer& hypotheses, std::int64_t channel_count)
{
    std::vector<std::string> channels;
    if (hypotheses.has_metadata("payload.leakage.channels")) {
        channels = split_comma_list(hypotheses.metadata("payload.leakage.channels"));
        if (channels.size() != static_cast<std::size_t>(channel_count)) {
            throw std::invalid_argument("DpaAttack payload.leakage.channels metadata must match hypothesis channel axis");
        }
        return channels;
    }

    channels.reserve(static_cast<std::size_t>(channel_count));
    for (std::int64_t index = 0; index < channel_count; ++index) {
        channels.push_back("channel_" + std::to_string(index));
    }
    return channels;
}

[[nodiscard]] std::vector<std::int64_t> unit_indexes_from_metadata(const Buffer& hypotheses, std::int64_t unit_count)
{
    if (hypotheses.has_metadata("attack.unit.indexes")) {
        auto indexes = parse_int_list(hypotheses.metadata("attack.unit.indexes"));
        if (indexes.size() != static_cast<std::size_t>(unit_count)) {
            throw std::invalid_argument("DpaAttack attack.unit.indexes metadata must match hypothesis unit axis");
        }
        return indexes;
    }
    if (hypotheses.has_metadata("payload.leakage.byte_indexes")) {
        auto indexes = parse_int_list(hypotheses.metadata("payload.leakage.byte_indexes"));
        if (indexes.size() != static_cast<std::size_t>(unit_count)) {
            throw std::invalid_argument("DpaAttack payload.leakage.byte_indexes metadata must match hypothesis unit axis");
        }
        return indexes;
    }

    std::vector<std::int64_t> indexes;
    indexes.reserve(static_cast<std::size_t>(unit_count));
    for (std::int64_t index = 0; index < unit_count; ++index) {
        indexes.push_back(index);
    }
    return indexes;
}

[[nodiscard]] torch::Tensor default_guess_values(std::int64_t guess_count, const torch::Device& device)
{
    return torch::arange(0, guess_count, torch::TensorOptions().dtype(torch::kLong).device(device));
}

[[nodiscard]] torch::Tensor guess_values_from_metadata(
    const Buffer& hypotheses,
    std::int64_t guess_count,
    const torch::Device& device)
{
    if (!hypotheses.has_metadata("attack.guess.values")) {
        return default_guess_values(guess_count, device);
    }

    const auto text = trim_to_string(hypotheses.metadata("attack.guess.values"));
    const auto range_pos = text.find("..");
    std::vector<std::int64_t> values;
    if (range_pos != std::string::npos) {
        const auto begin_text = trim_to_string(std::string_view(text).substr(0, range_pos));
        const auto end_text = trim_to_string(std::string_view(text).substr(range_pos + 2));
        const auto begin = std::stoll(begin_text);
        const auto end = std::stoll(end_text);
        if (end < begin) {
            throw std::invalid_argument("DpaAttack attack.guess.values range must be ascending");
        }
        values.reserve(static_cast<std::size_t>(end - begin + 1));
        for (auto value = begin; value <= end; ++value) {
            values.push_back(value);
        }
    } else {
        values = parse_int_list(text);
    }

    if (values.size() != static_cast<std::size_t>(guess_count)) {
        throw std::invalid_argument("DpaAttack attack.guess.values metadata must match hypothesis guess axis");
    }
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU)).to(device);
}

[[nodiscard]] std::vector<bool> channel_selection(
    const Buffer& hypotheses,
    std::int64_t channel_count,
    ScoreChannels score_channels)
{
    std::vector<bool> selected(static_cast<std::size_t>(channel_count), true);
    if (score_channels == ScoreChannels::All) {
        return selected;
    }

    if (!hypotheses.has_metadata("attack.channel.depends_on_guess")) {
        return selected;
    }

    const auto values = split_comma_list(hypotheses.metadata("attack.channel.depends_on_guess"));
    if (values.size() != static_cast<std::size_t>(channel_count)) {
        throw std::invalid_argument(
            "DpaAttack attack.channel.depends_on_guess metadata must match hypothesis channel axis");
    }
    bool any = false;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto lowered = lower_string(values[index]);
        if (lowered == "true") {
            selected[index] = true;
            any = true;
        } else if (lowered == "false") {
            selected[index] = false;
        } else {
            throw std::invalid_argument("DpaAttack attack.channel.depends_on_guess values must be true or false");
        }
    }
    if (!any) {
        throw std::invalid_argument("DpaAttack score_channels=guess_dependent selected no channels");
    }
    return selected;
}

[[nodiscard]] torch::Tensor channel_mask_tensor(
    const std::vector<bool>& selected,
    const torch::Device& device)
{
    std::vector<std::uint8_t> mask_values;
    mask_values.reserve(selected.size());
    for (const auto value : selected) {
        mask_values.push_back(value ? std::uint8_t{1} : std::uint8_t{0});
    }
    return torch::tensor(mask_values, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
        .to(torch::kBool)
        .to(device)
        .reshape({1, 1, static_cast<std::int64_t>(selected.size()), 1});
}

[[nodiscard]] ScoreResult score_difference(
    const torch::Tensor& difference,
    ScoreMethod score_method,
    const std::vector<bool>& selected_channels)
{
    const auto unit_count = difference.size(0);
    const auto guess_count = difference.size(1);
    const auto channel_count = difference.size(2);
    const auto sample_count = difference.size(3);

    torch::Tensor metric;
    switch (score_method) {
    case ScoreMethod::MaxAbs:
        metric = difference.abs();
        break;
    case ScoreMethod::MaxPositive:
        metric = difference;
        break;
    case ScoreMethod::MaxNegative:
        metric = -difference;
        break;
    }

    const auto mask = channel_mask_tensor(selected_channels, difference.device());
    const auto excluded = torch::full_like(metric, -std::numeric_limits<double>::infinity());
    metric = torch::where(mask, metric, excluded);

    const auto flat = metric.reshape({unit_count, guess_count, channel_count * sample_count});
    const auto max_result = flat.max(2);
    auto scores = std::get<0>(max_result).contiguous();
    auto flat_indexes = std::get<1>(max_result).to(torch::kLong).contiguous();

    const auto sort_result = scores.sort(1, true);
    auto ranking = std::get<1>(sort_result).to(torch::kLong).contiguous();
    auto best_guess_index = ranking.select(1, 0).contiguous();
    auto gather_index = best_guess_index.unsqueeze(1);
    auto best_score = scores.gather(1, gather_index).squeeze(1).contiguous();
    auto best_flat_index = flat_indexes.gather(1, gather_index).squeeze(1).to(torch::kLong).contiguous();
    auto best_channel = torch::floor_divide(best_flat_index, sample_count).to(torch::kLong).contiguous();
    auto best_sample = torch::remainder(best_flat_index, sample_count).to(torch::kLong).contiguous();

    return ScoreResult{
        .scores = std::move(scores),
        .ranking = std::move(ranking),
        .best_guess_index = std::move(best_guess_index),
        .best_score = std::move(best_score),
        .best_channel = std::move(best_channel),
        .best_sample = std::move(best_sample),
    };
}

void copy_hypothesis_semantic_metadata(const Buffer& hypotheses, Buffer& output)
{
    static const std::vector<std::string_view> keys{
        "payload.leakage.model",
        "payload.leakage.hypothesis",
        "payload.leakage.byte_indexes",
        "payload.leakage.channels",
        "payload.crypto.algorithm",
        "payload.crypto.state_bytes",
        "attack.hypothesis.algorithm",
        "attack.hypothesis.round",
        "attack.unit.kind",
        "attack.unit.indexes",
        "attack.guess.kind",
        "attack.guess.count",
        "attack.guess.order",
        "attack.guess.values",
        "attack.channel.depends_on_guess",
    };
    for (const auto key : keys) {
        if (hypotheses.has_metadata(key)) {
            output.set_metadata(std::string(key), hypotheses.metadata(key));
        }
    }
}

} // namespace

struct DpaAttack::IncrementalState {
    DifferenceSums sums;
    bool initialized = false;
    std::int64_t unit_count = 0;
    std::int64_t guess_count = 0;
    std::int64_t channel_count = 0;
    std::int64_t sample_count = 0;
    torch::ScalarType dtype = torch::kFloat32;
    torch::Device device = torch::kCPU;

    void reset()
    {
        sums = DifferenceSums{};
        initialized = false;
        unit_count = 0;
        guess_count = 0;
        channel_count = 0;
        sample_count = 0;
        dtype = torch::kFloat32;
        device = torch::Device(torch::kCPU);
    }

    torch::Tensor update(const PreparedInputs& prepared)
    {
        const auto batch = difference_sums(prepared);
        if (!initialized) {
            sums = batch;
            unit_count = prepared.unit_count;
            guess_count = prepared.guess_count;
            channel_count = prepared.channel_count;
            sample_count = prepared.sample_count;
            dtype = prepared.dtype;
            device = prepared.features.device();
            initialized = true;
            return difference_from_sums(sums);
        }

        if (prepared.unit_count != unit_count || prepared.guess_count != guess_count
            || prepared.channel_count != channel_count || prepared.sample_count != sample_count) {
            throw std::invalid_argument("DpaAttack incremental updates must keep [U,G,L,S] shape");
        }
        if (prepared.dtype != dtype) {
            throw std::invalid_argument("DpaAttack incremental updates must keep compute dtype");
        }
        if (prepared.features.device() != device) {
            throw std::invalid_argument("DpaAttack incremental updates must stay on the same device");
        }
        if (sums.count > std::numeric_limits<std::int64_t>::max() - batch.count) {
            throw std::overflow_error("DpaAttack incremental observation count overflow");
        }

        sums.count += batch.count;
        sums.sum_x = sums.sum_x + batch.sum_x;
        sums.group1_count = sums.group1_count + batch.group1_count;
        sums.group1_sum = sums.group1_sum + batch.group1_sum;
        return difference_from_sums(sums);
    }

    [[nodiscard]] std::int64_t observation_count() const
    {
        return initialized ? sums.count : 0;
    }
};

DpaAttack::DpaAttack(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

DpaAttack::~DpaAttack() = default;

ElementDescriptor DpaAttack::descriptor()
{
    return {
        .type_name = "DpaAttack",
        .klass = "Analyze/SCA/Attack/DPA",
        .purpose = "rank key guesses by difference of means between binary hypothesis groups",
        .input_pads = {
            Pad("features", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
            Pad("hypotheses", PadDirection::Input, torch_tensor_caps({{leakflow::base::caps_param_rank, "4"}})),
        },
        .output_pads = {
            Pad("scores", PadDirection::Output, Caps(attack_scores_caps_type)),
        },
        .property_specs = {
            PropertySpec(
                "score_method",
                std::string("max_abs"),
                "score reduction over difference-of-means [L,S]",
                "",
                StringEnumConstraint{{"max_abs", "max_positive", "max_negative"}},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"scores"}}),
            PropertySpec(
                "score_channels",
                std::string("guess_dependent"),
                "which binary hypothesis channels participate in scoring",
                "",
                StringEnumConstraint{{"guess_dependent", "all"}},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"scores"}}),
            PropertySpec(
                "top_k",
                std::int64_t{5},
                "number of top units/guesses shown by summaries and converters",
                "",
                IntRangeConstraint{1, 1024},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"scores"}}),
            PropertySpec(
                "accumulation_mode",
                std::string("auto"),
                "DPA accumulation mode: auto follows upstream liveness; recompute uses only the current buffer; "
                "incremental accumulates group sums across buffers",
                "",
                StringEnumConstraint{{"auto", "recompute", "incremental"}},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"scores"}}),
            PropertySpec(
                "active_accumulation_mode",
                std::string("recompute"),
                "Resolved accumulation mode currently selected by accumulation_mode",
                "",
                StringEnumConstraint{{"recompute", "incremental"}},
                "",
                PropertyEffect{},
                false,
                false),
            PropertySpec(
                "compute_dtype",
                std::string("input"),
                "DPA difference-of-means compute dtype",
                "",
                StringEnumConstraint{{"input", "float32", "float64"}},
                "",
                PropertyEffect{.kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"scores"}}),
        },
        .telemetry_specs = {
            make_duration_telemetry_spec("prepare", "time marshaling/casting/reshaping the feature and hypothesis tensors"),
            make_duration_telemetry_spec("difference", "time accumulating binary groups and computing difference of means"),
            make_duration_telemetry_spec("score", "time scoring and ranking the difference of means into key-guess scores"),
        },
        .keywords = {"dpa", "attack", "difference-of-means", "sca"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor("routing.element", std::string(),
                "element instance name that produced the DPA attack buffer", {"attack"}),
            make_element_metadata_descriptor("attack.method", std::string(), "attack method", {dpa_attack_method_id}),
            make_element_metadata_descriptor(
                "attack.difference.method", std::string(), "difference method", {"mean(group1)-mean(group0)"}),
            make_element_metadata_descriptor(
                "attack.accumulation.mode", std::string(), "effective accumulation mode", {"recompute", "incremental"}),
            make_element_metadata_descriptor(
                "attack.observation_count", std::int64_t{}, "number of observations represented by the attack", {"50"}),
            make_element_metadata_descriptor(
                "attack.feature.count", std::int64_t{}, "number of trace samples/features", {"5000"}),
            make_element_metadata_descriptor(
                "attack.score.method", std::string(), "score reduction method", {"max_abs"}),
            make_element_metadata_descriptor(
                "attack.score.channels", std::string(), "channels used for scoring", {"guess_dependent", "all"}),
            make_element_metadata_descriptor(
                "attack.score.order", std::string(), "score tensor order", {"unsorted"}),
            make_element_metadata_descriptor(
                "attack.ranking.order", std::string(), "ranking tensor order", {"descending_score"}),
            make_element_metadata_descriptor(
                "attack.ranking.values", std::string(), "ranking tensor value meaning", {"guess_index"}),
        },
    };
}

void DpaAttack::start()
{
    update_active_accumulation_mode();
    reset_incremental_state();
}

std::optional<Buffer> DpaAttack::process(std::optional<Buffer>)
{
    throw std::invalid_argument("DpaAttack requires named features and hypotheses inputs");
}

std::optional<Buffer> DpaAttack::process_inputs(ElementInputs inputs)
{
    const auto& features_buffer = required_input(inputs, "features", "DpaAttack");
    const auto& hypotheses_buffer = required_input(inputs, "hypotheses", "DpaAttack");
    const auto features_payload = torch_payload_for(features_buffer, "features");
    const auto hypotheses_payload = torch_payload_for(hypotheses_buffer, "hypotheses");

    const auto compute_dtype_text = string_property_or(*this, "compute_dtype", "input");
    const auto compute_dtype = compute_dtype_for(compute_dtype_text);
    const auto prepared = [&] {
        const auto scope = profile_scope("prepare");
        return prepare_inputs(features_payload->tensor(), hypotheses_payload->tensor(), compute_dtype);
    }();

    update_active_accumulation_mode();
    const auto active_mode = accumulation_mode_for(string_property_or(*this, "active_accumulation_mode", "recompute"));
    torch::Tensor difference;
    std::int64_t observation_count = prepared.trace_count;

    {
        const auto scope = profile_scope("difference");
        if (active_mode == AccumulationMode::Recompute) {
            reset_incremental_state();
            difference = difference_from_sums(difference_sums(prepared));
        } else {
            const auto options_changed =
                !incremental_compute_dtype_ || *incremental_compute_dtype_ != compute_dtype_text;
            if (!incremental_state_ || options_changed) {
                incremental_state_ = std::make_unique<IncrementalState>();
                incremental_compute_dtype_ = compute_dtype_text;
            }
            difference = incremental_state_->update(prepared);
            observation_count = incremental_state_->observation_count();
        }
    }

    const auto score_method_text = string_property_or(*this, "score_method", "max_abs");
    const auto score_channels_text = string_property_or(*this, "score_channels", "guess_dependent");
    const auto selected_channels =
        channel_selection(hypotheses_buffer, prepared.channel_count, score_channels_for(score_channels_text));
    auto score_result = [&] {
        const auto scope = profile_scope("score");
        return score_difference(difference, score_method_for(score_method_text), selected_channels);
    }();
    auto guess_values = guess_values_from_metadata(hypotheses_buffer, prepared.guess_count, difference.device());
    auto best_guess =
        guess_values.index_select(0, score_result.best_guess_index.to(guess_values.device())).to(torch::kLong).contiguous();
    const auto channels = channels_from_metadata(hypotheses_buffer, prepared.channel_count);
    const auto unit_indexes = unit_indexes_from_metadata(hypotheses_buffer, prepared.unit_count);
    const auto top_k = integer_property_or(*this, "top_k", 5);

    auto payload = std::make_shared<AttackScoresPayload>(
        score_result.scores,
        score_result.ranking,
        best_guess,
        score_result.best_guess_index,
        score_result.best_score,
        score_result.best_channel,
        score_result.best_sample,
        guess_values,
        std::nullopt,
        unit_indexes,
        channels,
        score_method_text,
        score_channels_text,
        observation_count,
        top_k);

    Buffer output{Caps(attack_scores_caps_type)};
    forward_metadata(inputs, profile_for_klass(element_kclass()), output, name());
    copy_hypothesis_semantic_metadata(hypotheses_buffer, output);
    output.set_metadata("routing.element", name());
    output.set_metadata("attack.method", dpa_attack_method_id);
    output.set_metadata("attack.difference.method", "mean(group1)-mean(group0)");
    output.set_metadata(
        "attack.accumulation.mode",
        active_mode == AccumulationMode::Recompute ? "recompute" : "incremental");
    output.set_metadata("attack.observation_count", std::to_string(observation_count));
    output.set_metadata("attack.feature.count", std::to_string(prepared.sample_count));
    output.set_metadata("attack.score.method", score_method_text);
    output.set_metadata("attack.score.channels", score_channels_text);
    output.set_metadata("attack.score.order", "unsorted");
    output.set_metadata("attack.score.rank_order", "descending");
    output.set_metadata("attack.score.reduced_axes", "leakage_channel,sample");
    output.set_metadata("attack.ranking.order", "descending_score");
    output.set_metadata("attack.ranking.values", "guess_index");
    output.set_metadata("attack.best_guess.values", "guess_value");
    output.set_metadata("attack.differences.emitted", "false");
    output.set_payload(std::move(payload));

    auto record = make_log_record(log::LogLevel::Debug, "element", "computed DPA attack ranking");
    record.fields.emplace("attack.method", dpa_attack_method_id);
    record.fields.emplace("accumulation_mode", output.metadata("attack.accumulation.mode"));
    record.fields.emplace("score_method", score_method_text);
    record.fields.emplace("score_channels", score_channels_text);
    record.fields.emplace("units", std::to_string(prepared.unit_count));
    record.fields.emplace("guesses", std::to_string(prepared.guess_count));
    record.fields.emplace("samples", std::to_string(prepared.sample_count));
    record.fields.emplace("observations", std::to_string(observation_count));
    leakflow::log::write(std::move(record));

    return output;
}

bool DpaAttack::can_replay() const
{
    return string_property_or(*this, "active_accumulation_mode", "recompute") == "recompute";
}

void DpaAttack::reset_incremental_state()
{
    incremental_state_.reset();
    incremental_compute_dtype_.reset();
}

void DpaAttack::update_active_accumulation_mode()
{
    const auto configured_mode = accumulation_mode_for(string_property_or(*this, "accumulation_mode", "auto"));
    const auto active_mode = effective_accumulation_mode(configured_mode, is_live_driven());
    set_read_only_property(
        "active_accumulation_mode",
        std::string(active_mode == AccumulationMode::Recompute ? "recompute" : "incremental"));
}

void DpaAttack::property_changed(std::string_view name)
{
    if (name == "accumulation_mode") {
        update_active_accumulation_mode();
    }
}

void DpaAttack::live_driven_changed()
{
    update_active_accumulation_mode();
}

} // namespace leakflow::plugins::crypto
