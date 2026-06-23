#include "leakflow/plugins/base/descriptor_catalog.hpp"

#include "leakflow/plugins/base/fake_live_src.hpp"
#include "leakflow/plugins/base/torch_convert.hpp"
#include "leakflow/plugins/base/torch_file_sink.hpp"
#include "leakflow/plugins/base/torch_file_src.hpp"
#include "base_plugin_constants.hpp"

#include <memory>
#include <utility>

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
            {"TorchConvert", [](std::string name) { return std::make_shared<TorchConvert>(std::move(name)); }},
            {"TorchFileSink", [](std::string name) { return std::make_shared<TorchFileSink>(std::move(name)); }},
        });
}

} // namespace leakflow::plugins::base
