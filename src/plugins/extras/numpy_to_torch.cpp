#include "leakflow/plugins/extras/numpy_to_torch.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/extras/numpy_payload.hpp"
#include "leakflow/extras/numpy_to_torch.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>

namespace leakflow::plugins::extras {
namespace {

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }

    return fallback;
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

[[nodiscard]] std::optional<torch::ScalarType> dtype_for_property(std::string_view dtype)
{
    const auto lowered = lower_string(dtype);
    if (lowered == "preserve") {
        return std::nullopt;
    }
    if (lowered == "bool") {
        return torch::kBool;
    }
    if (lowered == "int8") {
        return torch::kInt8;
    }
    if (lowered == "int16") {
        return torch::kInt16;
    }
    if (lowered == "int32") {
        return torch::kInt32;
    }
    if (lowered == "int64") {
        return torch::kInt64;
    }
    if (lowered == "uint8") {
        return torch::kUInt8;
    }
    if (lowered == "uint16") {
        return torch::kUInt16;
    }
    if (lowered == "uint32") {
        return torch::kUInt32;
    }
    if (lowered == "uint64") {
        return torch::kUInt64;
    }
    if (lowered == "float32") {
        return torch::kFloat32;
    }
    if (lowered == "float64") {
        return torch::kFloat64;
    }

    throw std::invalid_argument("NumpyToTorch dtype property is not supported");
}

[[nodiscard]] leakflow::extras::NumpyToTorchOptions conversion_options_for(const Element& element)
{
    leakflow::extras::NumpyToTorchOptions options;
    options.target_dtype = dtype_for_property(string_property_or(element, "dtype", "preserve"));
    options.target_device = torch::Device(string_property_or(element, "device", "cpu"));
    return options;
}

} // namespace

ElementDescriptor NumpyToTorch::descriptor()
{
    return {
        .type_name = "NumpyToTorch",
        .klass = "Convert/Tensor/NumpyToTorch",
        .purpose = "convert a NumpyPayload into a TorchTensorPayload using conversion id numpy-to-torch",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(leakflow::extras::numpy_array_caps_type)),
        },
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .property_specs = {
            PropertySpec(
                "dtype",
                std::string("preserve"),
                "target Torch dtype or preserve",
                "",
                StringEnumConstraint{{"preserve",
                    "bool",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "float32",
                    "float64"}}),
            PropertySpec("device", std::string("cpu"), "target Torch device such as cpu or cuda:0"),
        },
        .keywords = {"numpy", "torch", "conversion", numpy_to_torch_conversion_id, "extras"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.conversion.id",
                std::string(),
                "conversion implementation identifier",
                {numpy_to_torch_conversion_id}),
            make_element_metadata_descriptor(
                "payload.conversion.element",
                std::string(),
                "element instance name that performed the conversion",
                {"numpytotorch0"}),
            make_element_metadata_descriptor(
                "payload.layout",
                std::string(),
                "ordered axes preserved from the NumPy input",
                {"trace/sample", "axis_0/axis_1"}),
        },
    };
}

NumpyToTorch::NumpyToTorch(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> NumpyToTorch::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("NumpyToTorch requires an input buffer");
    }
    if (input->caps().type() != leakflow::extras::numpy_array_caps_type) {
        throw std::invalid_argument("NumpyToTorch requires leakflow/numpy-array input caps");
    }

    const auto numpy_payload = input->payload_as<leakflow::extras::NumpyPayload>();
    if (!numpy_payload) {
        throw std::invalid_argument("NumpyToTorch requires a NumpyPayload");
    }

    auto torch_payload = leakflow::extras::convert_numpy_to_torch(*numpy_payload, conversion_options_for(*this));

    Buffer output{torch_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "sink", name());
    output.set_metadata("payload.conversion.id", numpy_to_torch_conversion_id);
    output.set_metadata("payload.conversion.element", name());
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(torch_payload)));
    if (input->has_metadata("payload.layout")) {
        output.set_metadata("payload.layout", input->metadata("payload.layout"));
    }

    auto record = make_log_record(log::LogLevel::Debug, "element", "converted NumPy payload to Torch tensor");
    record.fields.emplace("payload.conversion.id", numpy_to_torch_conversion_id);
    record.fields.emplace("caps", output.caps().to_string());
    leakflow::log::write(std::move(record));
    return output;
}

} // namespace leakflow::plugins::extras
