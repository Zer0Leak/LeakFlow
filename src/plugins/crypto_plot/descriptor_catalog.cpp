#include "leakflow/plugins/crypto_plot/descriptor_catalog.hpp"

#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plot/score_view.hpp"
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

    // One ScoreView shared by every ScorePlot in this run: it is the score registry
    // (all units/elements stack in one group window) and is registered with the
    // PlotRuntime so the UI draws and clears it alongside the trace plots.
    auto view = std::make_shared<leakflow::plot::ScoreView>();
    runtime->add_view(view);

    registry.register_plugin(
        plugin_descriptors().front(),
        {
            {"ScorePlot", [view = std::move(view)](std::string name) {
                 return std::make_shared<ScorePlot>(view, std::move(name));
             }},
        });
}

} // namespace leakflow::plugins::crypto_plot
