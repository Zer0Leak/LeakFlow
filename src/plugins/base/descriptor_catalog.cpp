#include "leakflow/plugins/base/descriptor_catalog.hpp"

#include "leakflow/base/buffer_archive.hpp"
#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plugins/base/app_src.hpp"
#include "leakflow/plugins/base/fake_live_src.hpp"
#include "leakflow/plugins/base/torch_convert.hpp"
#include "leakflow/plugins/base/torch_file_sink.hpp"
#include "leakflow/plugins/base/torch_file_src.hpp"
#include "base_plugin_constants.hpp"

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


void register_payload_codecs(PayloadCodecRegistry& codecs)
{
    codecs.register_codec(
        leakflow::base::torch_tensor_caps_type,
        PayloadCodec{
            .save =
                [](const Payload& payload, leakflow::base::BufferArchiveWriter& archive) {
                    const auto* tensor_payload = dynamic_cast<const leakflow::base::TorchTensorPayload*>(&payload);
                    if (tensor_payload == nullptr) {
                        throw std::invalid_argument("torch-tensor codec: payload is not a TorchTensorPayload");
                    }
                    archive.write_tensor("tensor", tensor_payload->tensor());
                },
            .load =
                [](const leakflow::base::BufferArchiveReader& archive) -> std::shared_ptr<Payload> {
                    return std::make_shared<leakflow::base::TorchTensorPayload>(archive.read_tensor("tensor"));
                },
        });
}

} // namespace leakflow::plugins::base
