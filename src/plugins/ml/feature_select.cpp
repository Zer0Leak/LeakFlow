#include "leakflow/plugins/ml/feature_select.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>

namespace leakflow::plugins::ml {
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

} // namespace

ElementDescriptor FeatureSelect::descriptor()
{
    return {
        .type_name = "FeatureSelect",
        .klass = "Transform/Feature/Select",
        .purpose = "gather feature columns by index (e.g. truncate traces to PoI columns per unit)",
        .input_pads = {
            Pad("features", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
            // The indexes are the PoI sample positions selected on a (profiling)
            // capture and applied here to select columns of the current features, so
            // this pad is a Reference: its dataset identity forwards as provenance
            // (origin.indexes.*), not as the selected features' capture identity.
            Pad("indexes", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type), PadPresence::Required,
                PadMetadataRole::Reference),
        },
        .output_pads = {
            Pad("selected", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .keywords = {"feature", "select", "gather", "poi", "truncate", "ml"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.layout", std::string(), "logical axes of the selected feature tensor",
                {"observation/feature", "unit/observation/feature"}),
        },
    };
}

FeatureSelect::FeatureSelect(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> FeatureSelect::process(std::optional<Buffer>)
{
    throw std::invalid_argument("FeatureSelect requires named features and indexes inputs");
}

std::optional<Buffer> FeatureSelect::process_inputs(ElementInputs inputs)
{
    const auto& features_buffer = required_input(inputs, "features", "FeatureSelect");
    const auto& indexes_buffer = required_input(inputs, "indexes", "FeatureSelect");
    const auto features = tensor_input(features_buffer, "features");
    const auto indexes = tensor_input(indexes_buffer, "indexes").to(torch::kLong);

    torch::Tensor selected;
    if (indexes.dim() == 1) {
        // Shared columns for every unit; works for [T, N] and [U, T, N].
        selected = features.index_select(features.dim() - 1, indexes).contiguous();
    } else if (indexes.dim() == 2) {
        const auto units = indexes.size(0);
        const auto n_sel = indexes.size(1);
        torch::Tensor batched;
        if (features.dim() == 2) {
            batched = features.unsqueeze(0).expand({units, features.size(0), features.size(1)});
        } else if (features.dim() == 3) {
            if (features.size(0) != units) {
                throw std::invalid_argument("FeatureSelect: features unit axis must match indexes unit axis");
            }
            batched = features;
        } else {
            throw std::invalid_argument("FeatureSelect: features must be [T, N] or [U, T, N]");
        }
        const auto index = indexes.unsqueeze(1).expand({units, batched.size(1), n_sel});
        selected = batched.gather(2, index).contiguous(); // [U, T, N_sel]
    } else {
        throw std::invalid_argument("FeatureSelect: indexes must be [N_sel] or [U, N_sel]");
    }

    auto payload = leakflow::base::TorchTensorPayload(selected);
    Buffer output{payload.caps()};
    forward_metadata(*this, inputs, output);
    output.set_metadata("payload.feature.selected_count", std::to_string(selected.size(-1)));
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    output.set_metadata("payload.layout",
        selected.dim() == 2 ? "observation/feature" : "unit/observation/feature");
    return output;
}

} // namespace leakflow::plugins::ml
