#include "leakflow/plugins/core/descriptor_catalog.hpp"

#include "leakflow/plugins/core/fake_sink.hpp"
#include "leakflow/plugins/core/fake_src.hpp"
#include "leakflow/plugins/core/file_sink.hpp"
#include "leakflow/plugins/core/file_src.hpp"
#include "leakflow/plugins/core/queue.hpp"
#include "leakflow/plugins/core/summary.hpp"
#include "leakflow/plugins/core/sync.hpp"
#include "leakflow/plugins/core/tee.hpp"

#include <memory>
#include <utility>

namespace leakflow::plugins::core {
namespace {

constexpr auto core_author = "Zer0Leak <edgard.lima@gmail.com>";
constexpr auto core_license = "Apache-2.0";

} // namespace

std::vector<PluginDescriptor> plugin_descriptors()
{
    return {
        with_common_element_properties(PluginDescriptor{
            .name = plugin_name,
            .owner = core_author,
            .author = core_author,
            .license = core_license,
            .version = "0.10",
            .purpose = "shared library with generic core pipeline elements",
            .keywords = {"core", "file", "fake", "summary", "tee", "queue"},
            .elements = {
                FileSrc::descriptor(),
                FileSink::descriptor(),
                FakeSrc::descriptor(),
                FakeSink::descriptor(),
                Summary::descriptor(),
                Tee::descriptor(),
                Queue::descriptor(),
                Sync::descriptor(),
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
            {"FileSrc", [](std::string name) { return std::make_shared<FileSrc>(std::move(name)); }},
            {"FileSink", [](std::string name) { return std::make_shared<FileSink>(std::move(name)); }},
            {"FakeSrc", [](std::string name) { return std::make_shared<FakeSrc>(std::move(name)); }},
            {"FakeSink", [](std::string name) { return std::make_shared<FakeSink>(std::move(name)); }},
            {"Summary", [](std::string name) { return std::make_shared<Summary>(std::move(name)); }},
            {"Tee", [](std::string name) { return std::make_shared<Tee>(std::move(name)); }},
            {"Queue", [](std::string name) { return std::make_shared<Queue>(std::move(name)); }},
            {"Sync", [](std::string name) { return std::make_shared<Sync>(std::move(name)); }},
        });
}

} // namespace leakflow::plugins::core
