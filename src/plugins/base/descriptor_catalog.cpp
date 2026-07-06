#include "leakflow/plugins/base/descriptor_catalog.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plugins/base/app_src.hpp"
#include "leakflow/plugins/base/fake_live_src.hpp"
#include "leakflow/plugins/base/torch_convert.hpp"
#include "leakflow/plugins/base/torch_file_sink.hpp"
#include "leakflow/plugins/base/torch_file_src.hpp"
#include "base_plugin_constants.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::base {

std::vector<PluginDescriptor> plugin_descriptors()
{
    return {
        with_common_element_properties(PluginDescriptor{
            .name = plugin_name,
            .owner = base_author,
            .author = base_author,
            .license = base_license,
            .version = base_version,
            .purpose = "Leakflow base elements",
            .keywords = {"base", "torch", "pt", "source", "sink", "conversion"},
            .elements = {
                TorchFileSrc::descriptor(),
                TorchConvert::descriptor(),
                TorchFileSink::descriptor(),
                FakeLiveSrc::descriptor(),
                AppSrc::descriptor(),
            },
        }),
    };
}

const PluginDescriptor* find_plugin_descriptor(std::string_view name)
{
    static const auto descriptors = plugin_descriptors();

    for (const auto& descriptor : descriptors) {
        if (descriptor.name == name) {
            return &descriptor;
        }
    }

    return nullptr;
}

void register_plugin_descriptors(DescriptorRegistry& registry)
{
    registry.register_plugins(plugin_descriptors());
}

void register_element_factories(ElementFactoryRegistry& registry)
{
    registry.register_plugin(
        plugin_descriptors().front(),
        {
            {"TorchFileSrc", [](std::string name) { return std::make_shared<TorchFileSrc>(std::move(name)); }},
            {"FakeLiveSrc", [](std::string name) { return std::make_shared<FakeLiveSrc>(std::move(name)); }},
            {"AppSrc", [](std::string name) { return std::make_shared<AppSrc>(std::move(name)); }},
            {"TorchConvert", [](std::string name) { return std::make_shared<TorchConvert>(std::move(name)); }},
            {"TorchFileSink", [](std::string name) { return std::make_shared<TorchFileSink>(std::move(name)); }},
        });
}

namespace {

void write_blob(const std::filesystem::path& file, const std::vector<char>& data)
{
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("payload codec could not open for writing: " + file.string());
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) {
        throw std::runtime_error("payload codec failed while writing: " + file.string());
    }
}

[[nodiscard]] std::vector<char> read_blob(const std::filesystem::path& file)
{
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("payload codec could not open for reading: " + file.string());
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

void register_payload_codecs(PayloadCodecRegistry& codecs)
{
    codecs.register_codec(
        leakflow::base::torch_tensor_caps_type,
        PayloadCodec{
            .save =
                [](const Payload& payload, const std::filesystem::path& dir) {
                    const auto* tensor_payload = dynamic_cast<const leakflow::base::TorchTensorPayload*>(&payload);
                    if (tensor_payload == nullptr) {
                        throw std::invalid_argument("torch-tensor codec: payload is not a TorchTensorPayload");
                    }
                    write_blob(dir / "payload.pt", torch::pickle_save(torch::IValue(tensor_payload->tensor())));
                },
            .load =
                [](const std::filesystem::path& dir) -> std::shared_ptr<Payload> {
                    auto value = torch::pickle_load(read_blob(dir / "payload.pt"));
                    return std::make_shared<leakflow::base::TorchTensorPayload>(value.toTensor());
                },
        });
}

} // namespace leakflow::plugins::base
