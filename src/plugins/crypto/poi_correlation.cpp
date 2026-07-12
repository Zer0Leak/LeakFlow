#include "leakflow/plugins/crypto/poi_correlation.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/statistics.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include <cstdint>
#include <map>
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

[[nodiscard]] const Buffer& required_input(
    const ElementInputs& inputs, std::string_view pad, std::string_view element)
{
    const auto found = inputs.find(std::string(pad));
    if (found == inputs.end() || !found->second) {
        throw std::invalid_argument(std::string(element) + " requires connected input pad " + std::string(pad));
    }
    return *found->second;
}

[[nodiscard]] torch::Tensor tensor_input(const Buffer& buffer, std::string_view pad)
{
    if (buffer.caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument(std::string(pad) + " input must have leakflow/torch-tensor caps");
    }
    const auto payload = buffer.payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument(std::string(pad) + " input must carry a TorchTensorPayload");
    }
    return payload->tensor();
}

// The semantic unit index for each row of the target `[U, N, C]` tensor.
[[nodiscard]] std::vector<std::int64_t> target_unit_indexes(const Buffer& buffer)
{
    if (!buffer.has_metadata("attack.unit.indexes")) {
        return {};
    }
    auto text = buffer.metadata("attack.unit.indexes");
    const auto begin = text.find_first_not_of(" \t[");
    const auto end = text.find_last_not_of(" \t]");
    if (begin == std::string::npos) {
        return {};
    }
    text = text.substr(begin, end - begin + 1);
    std::vector<std::int64_t> units;
    std::size_t pos = 0;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        if (index == text.size() || text[index] == ',') {
            const auto token = text.substr(pos, index - pos);
            if (token.find_first_not_of(" \t") != std::string::npos) {
                units.push_back(std::stoll(token));
            }
            pos = index + 1;
        }
    }
    return units;
}

[[nodiscard]] std::string unit_indexes_metadata(const std::vector<CorrelationPoiResult>& results)
{
    auto value = std::string("[");
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index != 0) {
            value += ",";
        }
        value += std::to_string(results[index].unit_index);
    }
    value += "]";
    return value;
}

} // namespace

ElementDescriptor PoiCorrelation::descriptor()
{
    return {
        .type_name = "PoiCorrelation",
        .klass = "Analyze/SCA/PoiCorrelation",
        .purpose = "re-score PoI positions with their Pearson correlation on new traces",
        .input_pads = {
            Pad("poi", PadDirection::Input, Caps(correlation_poi_caps_type)),
            Pad("traces", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
            Pad("targets", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .output_pads = {
            Pad("poi", PadDirection::Output, Caps(correlation_poi_caps_type)),
        },
        .keywords = {"poi", "correlation", "re-score", "attack", "transfer", "sca"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.poi.rescored", std::string(), "PoI scores recomputed on the input traces", {"true"}),
            make_element_metadata_descriptor(
                "payload.layout", std::string(), "semantic payload layout",
                {"unit/channel/poi/[sample_index,correlation]"}),
            make_element_metadata_descriptor(
                "payload.poi.unit_count", std::int64_t{}, "number of units represented", {"16"}),
            make_element_metadata_descriptor(
                "attack.unit.count", std::int64_t{}, "number of attack units represented", {"16"}),
        },
    };
}

PoiCorrelation::PoiCorrelation(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> PoiCorrelation::process(std::optional<Buffer>)
{
    throw std::invalid_argument("PoiCorrelation requires named poi, traces, and targets inputs");
}

std::optional<Buffer> PoiCorrelation::process_inputs(ElementInputs inputs)
{
    const auto& poi_buffer = required_input(inputs, "poi", "PoiCorrelation");
    const auto& traces_buffer = required_input(inputs, "traces", "PoiCorrelation");
    const auto& targets_buffer = required_input(inputs, "targets", "PoiCorrelation");

    if (poi_buffer.caps().type() != correlation_poi_caps_type) {
        throw std::invalid_argument("PoiCorrelation poi input must have leakflow/correlation-poi caps");
    }
    const auto poi = poi_buffer.payload_as<CorrelationPoiPayload>();
    if (!poi) {
        throw std::invalid_argument("PoiCorrelation poi input must carry a CorrelationPoiPayload");
    }
    const auto traces = tensor_input(traces_buffer, "traces").to(torch::kFloat64); // [T, S]
    const auto leakage = tensor_input(targets_buffer, "targets").to(torch::kFloat64); // [U, N, C]
    if (traces.dim() != 2) {
        throw std::invalid_argument("PoiCorrelation traces must be [T, S]");
    }
    if (leakage.dim() != 3) {
        throw std::invalid_argument("PoiCorrelation targets must be a [U, N, C] tensor");
    }

    leakflow::base::PearsonCorrelationOptions options;
    options.compute_dtype = leakflow::base::PearsonComputeDtype::Float64;

    // Map each PoI unit to its target row. With attack.unit.indexes metadata we match by unit id;
    // without it, we fall back to positional
    // alignment. Either way we stay in range of the leakage tensor.
    const auto units = target_unit_indexes(targets_buffer);
    std::map<std::int64_t, std::int64_t> unit_to_row;
    for (std::size_t row = 0; row < units.size(); ++row) {
        unit_to_row.emplace(units[row], static_cast<std::int64_t>(row));
    }

    // Re-score each matching PoI (same positions) against the new leakage on the new traces.
    std::vector<CorrelationPoiResult> results;
    for (std::size_t index = 0; index < poi->results().size(); ++index) {
        const auto& original = poi->results()[index];
        std::int64_t row = static_cast<std::int64_t>(index);
        if (!unit_to_row.empty()) {
            const auto found = unit_to_row.find(static_cast<std::int64_t>(original.unit_index));
            if (found == unit_to_row.end()) {
                continue; // no leakage row for this unit -> drop it from the output
            }
            row = found->second;
        }
        if (row < 0 || row >= leakage.size(0)) {
            continue;
        }
        const auto rescored = original.result.clone(); // [channel, k, 2] = (index, score)
        const auto channel_count = rescored.size(0);
        for (std::int64_t channel = 0; channel < channel_count; ++channel) {
            const auto positions = rescored[channel].select(1, 0).to(torch::kLong); // [k]
            const auto features = traces.index_select(1, positions);                 // [T, k]
            const auto target = leakage[row].select(1, channel).unsqueeze(1);         // [T, 1]
            const auto correlation = leakflow::base::pearson_correlation(features, target, options); // [1, k]
            rescored[channel].select(1, 1).copy_(correlation.reshape({-1}));          // overwrite scores
        }
        results.push_back(CorrelationPoiResult{.unit_index = original.unit_index, .result = rescored});
    }
    if (results.empty()) {
        throw std::invalid_argument("PoiCorrelation: no PoI units match the target unit indexes");
    }

    const auto output_unit_indexes = unit_indexes_metadata(results);
    const auto output_unit_count = results.size();
    auto payload = std::make_shared<CorrelationPoiPayload>(std::move(results), poi->score_name());
    Buffer output{Caps(correlation_poi_caps_type)};
    forward_metadata(inputs, profile_for_klass(element_kclass()), output, name());
    output.set_metadata("payload.poi.rescored", "true");
    output.set_metadata("payload.poi.unit_count", std::to_string(output_unit_count));
    output.set_metadata("attack.unit.count", std::to_string(output_unit_count));
    output.set_metadata("attack.unit.indexes", output_unit_indexes);
    if (targets_buffer.has_metadata("attack.unit.kind")) {
        output.set_metadata("attack.unit.kind", targets_buffer.metadata("attack.unit.kind"));
    }
    output.set_payload(payload);
    return output;
}

} // namespace leakflow::plugins::crypto
