#include "leakflow/plugins/base/torch_convert.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>

namespace leakflow::plugins::base {
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

    throw std::invalid_argument("TorchConvert dtype property is not supported");
}

[[nodiscard]] std::optional<torch::Device> device_for_property(std::string_view device)
{
    const auto lowered = lower_string(device);
    if (lowered.empty() || lowered == "preserve") {
        return std::nullopt;
    }

    return torch::Device(std::string(device));
}

} // namespace

ElementDescriptor TorchConvert::descriptor()
{
    return {
        .type_name = "TorchConvert",
        .klass = "Convert/Tensor/Torch",
        .purpose = "convert a TorchTensorPayload to a requested Torch dtype or device",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
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
            PropertySpec("device", std::string("preserve"), "target Torch device such as cpu, cuda:0, or preserve"),
        },
        .keywords = {"torch", "conversion", torch_convert_conversion_id, "base"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.conversion.id",
                std::string(),
                "conversion implementation identifier",
                {torch_convert_conversion_id}),
            make_element_metadata_descriptor(
                "payload.conversion.element",
                std::string(),
                "element instance name that performed the conversion",
                {"torchconvert0"}),
        },
    };
}

TorchConvert::TorchConvert(std::string name) : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> TorchConvert::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("TorchConvert requires an input buffer");
    }
    if (input->caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument("TorchConvert requires leakflow/torch-tensor input caps");
    }

    const auto payload = input->payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument("TorchConvert requires a TorchTensorPayload");
    }

    auto tensor = payload->tensor();
    const auto target_dtype = dtype_for_property(string_property_or(*this, "dtype", "preserve"));
    const auto target_device = device_for_property(string_property_or(*this, "device", "preserve"));

    if (target_dtype) {
        tensor = tensor.to(*target_dtype);
    }
    if (target_device) {
        tensor = tensor.to(*target_device);
    }

    auto converted_payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    Buffer output{converted_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "sink", name());
    output.set_metadata("payload.conversion.id", torch_convert_conversion_id);
    output.set_metadata("payload.conversion.element", name());
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(converted_payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "converted Torch tensor");
    record.fields.emplace("payload.conversion.id", torch_convert_conversion_id);
    record.fields.emplace("caps", output.caps().to_string());
    leakflow::log::write(std::move(record));
    return output;
}

} // namespace leakflow::plugins::base
