#include "leakflow/plugins/base/torch_file_sink.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "base_plugin_constants.hpp"

#include <fstream>
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

} // namespace

ElementDescriptor TorchFileSink::descriptor()
{
    return {
        .type_name = "TorchFileSink",
        .klass = "Sink/File/Torch",
        .purpose = "write one TorchTensorPayload as a Torch tensor .pt file",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .output_pads = {},
        .property_specs = {
            PropertySpec("path", std::string(), "path to write one Torch tensor .pt file"),
        },
        .keywords = {"torch", "pt", "file", "sink", "base"},
    };
}

TorchFileSink::TorchFileSink(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> TorchFileSink::process(std::optional<Buffer> input)
{
    received_ = input.has_value();
    last_buffer_ = input;

    if (!input) {
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "received no buffer"));
        return std::nullopt;
    }
    if (input->caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument("TorchFileSink requires leakflow/torch-tensor input caps");
    }

    const auto payload = input->payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument("TorchFileSink requires a TorchTensorPayload");
    }

    const auto path = string_property_or(*this, "path", "");
    if (path.empty()) {
        throw std::invalid_argument("TorchFileSink path property must not be empty");
    }

    const auto data = torch::pickle_save(torch::IValue(payload->tensor()));
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("TorchFileSink could not open output path");
    }

    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!output) {
        throw std::runtime_error("TorchFileSink could not write output path");
    }

    auto record = make_log_record(log::LogLevel::Debug, "element", "wrote Torch tensor file");
    record.fields.emplace("bytes", std::to_string(data.size()));
    record.fields.emplace("path", path);
    leakflow::log::write(std::move(record));
    return std::nullopt;
}

bool TorchFileSink::received() const
{
    return received_;
}

const std::optional<Buffer>& TorchFileSink::last_buffer() const
{
    return last_buffer_;
}

} // namespace leakflow::plugins::base
