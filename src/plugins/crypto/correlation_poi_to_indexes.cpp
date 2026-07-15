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

// Result for a specific unit index, or throw if the payload has no such unit.
[[nodiscard]] const CorrelationPoiResult& result_for_unit(
    const CorrelationPoiPayload& payload, std::int64_t unit)
{
    for (const auto& result : payload.results()) {
        if (static_cast<std::int64_t>(result.unit_index) == unit) {
            return result;
        }
    }
    throw std::invalid_argument(
        "CorrelationPoiToIndexes: requested unit " + std::to_string(unit) + " is not in the PoI payload");
}

[[nodiscard]] torch::Tensor unit_indexes(const CorrelationPoiResult& result)
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
        .property_specs = {
            PropertySpec("units", Units::none(),
                "unit indexes to keep, e.g. [0] / [0:16] / [0,2:4]; none/[] = all, in payload order",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"indexes"},
                }),
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

    const auto units = property_as<Units>("units").value_or(Units::none()).to_vector();
    std::vector<torch::Tensor> per_unit;
    std::vector<std::int64_t> selected_units;
    if (units.empty()) {
        // All units, in payload order.
        per_unit.reserve(payload->results().size());
        selected_units.reserve(payload->results().size());
        for (const auto& result : payload->results()) {
            per_unit.push_back(unit_indexes(result));
            selected_units.push_back(result.unit_index);
        }
    } else {
        // Only the requested units, in the requested order.
        per_unit.reserve(units.size());
        selected_units.reserve(units.size());
        for (const auto unit : units) {
            per_unit.push_back(unit_indexes(result_for_unit(*payload, unit)));
            selected_units.push_back(unit);
        }
    }
    const auto indexes = torch::stack(per_unit, 0).contiguous(); // [len(units), N_sel]

    auto index_payload = leakflow::base::TorchTensorPayload(indexes);
    Buffer output{index_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "poi", name());
    copy_unit_metadata(*input, output);
    output.set_metadata("attack.unit.count", std::to_string(indexes.size(0)));
    output.set_metadata("attack.unit.indexes", unit_indexes_metadata(selected_units));
    output.set_metadata("payload.feature.selected_count", std::to_string(indexes.size(1)));
    // The PoI rows are per unit; label the leading axis so FeatureSelect and the
    // fusion downstream carry and check unit identity, not just shape.
    output.set_units(Units::of(selected_units));
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(index_payload)));
    output.set_metadata("payload.layout", "unit/feature");
    return output;
}

} // namespace leakflow::plugins::crypto
