#include "leakflow/plugins/crypto/correlation_poi_to_indexes.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

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

void copy_unit_metadata(const Buffer& input, Buffer& output)
{
    for (const auto key : {"attack.unit.kind"}) {
        if (input.has_metadata(key)) {
            output.set_metadata(key, input.metadata(key));
        }
    }
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

[[nodiscard]] torch::Tensor poi_sample_indexes(const CorrelationPoiResult& result)
{
    // result.result is [channel, k, 2] = (sample_index, score); take the indexes and
    // concatenate the channels -> [channel * k].
    return result.result.select(2, 0).to(torch::kLong).reshape({-1});
}

} // namespace

ElementDescriptor CorrelationPoiToIndexes::descriptor()
{
    return {
        .type_name = "CorrelationPoiToIndexes",
        .klass = "Convert/SCA/PoiIndexes",
        .purpose = "convert selected PoIs into a per-unit int index tensor [U, N_sel] for FeatureSelect",
        .input_pads = {
            Pad("poi", PadDirection::Input, Caps(correlation_poi_caps_type)),
        },
        .output_pads = {
            Pad("indexes", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .keywords = {"poi", "indexes", "feature", "select", "sca"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.feature.selected_count", std::int64_t{}, "PoI columns per unit", {"100"}),
            make_element_metadata_descriptor(
                "payload.layout", std::string(), "semantic payload layout", {"unit/feature"}),
            make_element_metadata_descriptor(
                "attack.unit.count", std::int64_t{}, "number of attack units represented", {"16"}),
        },
    };
}

CorrelationPoiToIndexes::CorrelationPoiToIndexes(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> CorrelationPoiToIndexes::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("CorrelationPoiToIndexes requires an input buffer");
    }
    if (input->caps().type() != correlation_poi_caps_type) {
        throw std::invalid_argument("CorrelationPoiToIndexes requires leakflow/correlation-poi input caps");
    }
    const auto payload = input->payload_as<CorrelationPoiPayload>();
    if (!payload) {
        throw std::invalid_argument("CorrelationPoiToIndexes requires a CorrelationPoiPayload");
    }
    if (payload->results().empty()) {
        throw std::invalid_argument("CorrelationPoiToIndexes: PoI payload has no units");
    }

    // Flatten every unit the payload carries, in payload order. Unit selection is an
    // upstream PoiSelect concern, not this converter's.
    std::vector<torch::Tensor> per_unit;
    std::vector<std::int64_t> selected_units;
    per_unit.reserve(payload->results().size());
    selected_units.reserve(payload->results().size());
    for (const auto& result : payload->results()) {
        per_unit.push_back(poi_sample_indexes(result));
        selected_units.push_back(result.unit_index);
    }
    const auto indexes = torch::stack(per_unit, 0).contiguous(); // [units, N_sel]

    auto index_payload = leakflow::base::TorchTensorPayload(indexes);
    Buffer output{index_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "poi", name());
    copy_unit_metadata(*input, output);
    output.set_metadata("attack.unit.count", std::to_string(indexes.size(0)));
    output.set_metadata("attack.unit.indexes", units_metadata(selected_units));
    output.set_metadata("payload.feature.selected_count", std::to_string(indexes.size(1)));
    // The PoI rows are per unit; label the leading axis so FeatureSelect and the
    // fusion downstream carry and check unit identity, not just shape.
    output.set_units(Units::of(selected_units));
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(index_payload)));
    output.set_metadata("payload.layout", "unit/feature");
    return output;
}

} // namespace leakflow::plugins::crypto
