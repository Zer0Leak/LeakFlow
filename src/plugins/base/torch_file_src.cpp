#include "leakflow/plugins/base/torch_file_src.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "base_plugin_constants.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::base {
namespace {

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] bool bool_property_or(const Element& element, std::string_view name, bool fallback)
{
    if (const auto value = element.property_as<bool>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] std::optional<torch::Device> load_device_for(const Element& element)
{
    const auto device = string_property_or(element, "device", "cpu");
    if (device.empty() || device == "preserve") {
        return std::nullopt;
    }

    return torch::Device(device);
}

[[nodiscard]] std::vector<char> read_binary_file(std::string_view path)
{
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("TorchFileSrc could not open input path");
    }

    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] std::optional<torch::Tensor> load_pickle_tensor(std::string_view path)
{
    try {
        const auto data = read_binary_file(path);
        const auto value = torch::pickle_load(data);
        if (!value.isTensor()) {
            throw std::invalid_argument("TorchFileSrc requires a serialized torch.Tensor");
        }

        return value.toTensor();
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

} // namespace

ElementDescriptor TorchFileSrc::descriptor()
{
    return {
        .type_name = "TorchFileSrc",
        .long_name = "Torch File Source",
        .rank = ElementRank::Primary,
        .klass = "Source/File/Torch",
        .purpose = "load one Torch tensor .pt file as a TorchTensorPayload",
        .input_pads = {},
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .property_specs = {
            // path/device/invert all change the produced payload (and path can
            // change its shape/dtype), so they are payload-output: a change reruns
            // the downstream path.
            PropertySpec("path", std::string(), "path to read one Torch tensor .pt file",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"},
                }),
            PropertySpec(
                "device",
                std::string("cpu"),
                "target Torch device for loaded tensor, or preserve",
                "",
                std::monostate{},
                "any Torch device string (examples: cpu, cuda, cuda:0), or preserve",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"},
                }),
            PropertySpec(
                "invert",
                false,
                "invert the loaded leakage tensor by multiplying it by -1",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"src"},
                }),
        },
        .keywords = {"torch", "pt", "source", "base"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "origin.file.format",
                std::string(),
                "source file format identifier",
                {"torch-tensor"},
                "Identifies TorchFileSrc output as a serialized single Torch tensor."),
            make_element_metadata_descriptor(
                "origin.file.path",
                std::string(),
                "input Torch tensor file path",
                {"traces/aes/sync/aes_sync_poi/key_01/traces.pt"}),
            make_element_metadata_descriptor(
                "origin.file.size",
                std::int64_t{},
                "input file size in bytes",
                {"40001506"}),
            make_element_metadata_descriptor(
                "payload.layout",
                std::string(),
                "ordered axes of the loaded tensor payload",
                {"axis_0/axis_1", "axis_0"}),
        },
        .metadata_suggestions = {
            make_element_metadata_descriptor(
                "capture.sample_rate_hz",
                0.0,
                "trace sample rate in samples per second",
                {"29454545.454545453"},
                "TracePlot uses this user metadata when x_axis=time_us and the sample_rate_hz property is zero."),
            make_element_metadata_descriptor(
                "capture.source",
                std::string(),
                "capture hardware, simulator, or acquisition source",
                {"ChipWhisperer"},
                "Useful for plot titles, reports, and reproducibility metadata."),
            make_element_metadata_descriptor(
                "capture.dataset.name",
                std::string(),
                "dataset or experiment identifier for downstream reporting",
                {"aes_sync_poi"}),
            make_element_metadata_descriptor(
                "payload.leakage.range",
                DoubleInterval{0.0, 0.0},
                "expected leakage value range",
                {"[-0.5,0.5]"},
                "Plotters and reports may use this to choose consistent y-axis bounds.",
                "[min,max]"),
            make_element_metadata_descriptor(
                "payload.leakage.inverted",
                false,
                "whether the leakage samples are inverted, as declared by the user",
                {"false", "true"},
                "TorchFileSrc does not set this automatically; the user records the final "
                "state based on the original data and the invert property.",
                "true or false"),
        },
    };
}

TorchFileSrc::TorchFileSrc(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> TorchFileSrc::process(std::optional<Buffer> input)
{
    if (input) {
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "forwarded existing input"));
        return input;
    }

    const auto path = string_property_or(*this, "path", "");
    if (path.empty()) {
        throw std::invalid_argument("TorchFileSrc path property must not be empty");
    }

    auto tensor = [&] {
        auto loaded = load_pickle_tensor(path);
        if (loaded) {
            if (const auto target_device = load_device_for(*this)) {
                return loaded->to(*target_device);
            }

            return *loaded;
        }

        torch::Tensor fallback;
        torch::load(fallback, path, load_device_for(*this));
        return fallback;
    }();

    const auto invert = bool_property_or(*this, "invert", false);
    if (invert) {
        if (!tensor.is_floating_point()) {
            throw std::invalid_argument("TorchFileSrc invert requires a floating-point tensor");
        }
        tensor = -tensor;
    }

    auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    Buffer buffer(payload.caps());
    buffer.set_metadata("origin.file.format", "torch-tensor");
    buffer.set_metadata("origin.file.path", path);
    buffer.set_metadata("origin.file.size", std::to_string(std::filesystem::file_size(path)));
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "loaded Torch tensor file");
    record.fields.emplace("caps", buffer.caps().to_string());
    record.fields.emplace("origin.file.format", buffer.metadata("origin.file.format"));
    record.fields.emplace("origin.file.path", path);
    record.fields.emplace("origin.file.size", buffer.metadata("origin.file.size"));
    record.fields.emplace("invert", invert ? "true" : "false");
    leakflow::log::write(std::move(record));
    return buffer;
}

} // namespace leakflow::plugins::base
