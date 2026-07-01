#include "leakflow/plugins/crypto_plot/descriptor_catalog.hpp"

#include "leakflow/plugins/crypto_plot/score_plot.hpp"
#include "crypto_plot_plugin_constants.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace leakflow::plugins::crypto_plot {

std::vector<PluginDescriptor> plugin_descriptors()
{
    return {
        with_common_element_properties(PluginDescriptor{
            .name = plugin_name,
            .owner = crypto_plot_author,
            .author = crypto_plot_author,
            .license = crypto_plot_license,
            .purpose = "shared library bridging crypto/SCA attack results to ImGui/ImPlot score plots",
            .keywords = {"plot", "score", "cpa", "attack", "sca"},
            .elements =
                {
                    ScorePlot::descriptor(),
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
        throw std::invalid_argument("ScorePlot factory registration requires a PlotRuntime");
    }

    registry.register_plugin(
        plugin_descriptors().front(),
        {
            {"ScorePlot", [runtime = std::move(runtime)](std::string name) {
                 return std::make_shared<ScorePlot>(runtime, std::move(name));
             }},
        });
}

} // namespace leakflow::plugins::crypto_plot
