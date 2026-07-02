#include "leakflow/plugins/plot/descriptor_catalog.hpp"

#include "leakflow/plugins/plot/trace_plot.hpp"
#include "plot_plugin_constants.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace leakflow::plugins::plot {

std::vector<PluginDescriptor> plugin_descriptors()
{
    return {
        with_common_element_properties(PluginDescriptor{
            .name = plugin_name,
            .owner = plot_author,
            .author = plot_author,
            .license = plot_license,
            .purpose = "shared library with ImGui/ImPlot pipeline elements",
            .keywords = {"plot", "imgui", "implot", "trace"},
            .elements =
                {
                    TracePlot::descriptor(),
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

void register_element_factories(
    ElementFactoryRegistry& registry,
    std::shared_ptr<leakflow::plot::PlotRuntime> runtime)
{
    if (!runtime) {
        throw std::invalid_argument("TracePlot factory registration requires a PlotRuntime");
    }

    registry.register_plugin(
        plugin_descriptors().front(),
        {
            {"TracePlot", [runtime = std::move(runtime)](std::string name) {
                 return std::make_shared<TracePlot>(runtime, std::move(name));
             }},
        });
}

} // namespace leakflow::plugins::plot
