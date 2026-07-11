#include "leakflow/plugins/extras/descriptor_catalog.hpp"

#include "leakflow/plugins/extras/fake_live_hdf5_src.hpp"
#include "leakflow/plugins/extras/hdf5_file_src.hpp"
#include "leakflow/plugins/extras/numpy_src.hpp"
#include "leakflow/plugins/extras/numpy_to_torch.hpp"

#include <memory>
#include <utility>

namespace leakflow::plugins::extras {
namespace {

constexpr auto extras_author = "Zer0Leak <edgard.lima@gmail.com>";
constexpr auto extras_license = "Apache-2.0";

} // namespace

std::vector<PluginDescriptor> plugin_descriptors()
{
    return {
        with_common_element_properties(PluginDescriptor{
            .name = plugin_name,
            .owner = extras_author,
            .author = extras_author,
            .license = extras_license,
            .purpose = "shared library with extras file-format, dataset, and conversion pipeline elements",
            .keywords = {"extras", "numpy", "npy", "hdf5", "h5", "source", "conversion", "torch"},
            .elements = {
                NumpySrc::descriptor(),
                NumpyToTorch::descriptor(),
                Hdf5FileSrc::descriptor(),
                FakeLiveHdf5Src::descriptor(),
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
            {"NumpySrc", [](std::string name) { return std::make_shared<NumpySrc>(std::move(name)); }},
            {"NumpyToTorch", [](std::string name) { return std::make_shared<NumpyToTorch>(std::move(name)); }},
            {"Hdf5FileSrc", [](std::string name) { return std::make_shared<Hdf5FileSrc>(std::move(name)); }},
            {"FakeLiveHdf5Src", [](std::string name) { return std::make_shared<FakeLiveHdf5Src>(std::move(name)); }},
        });
}

} // namespace leakflow::plugins::extras
