#include "leakflow/plugins/crypto_plot/descriptor_catalog.hpp"

#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plot/poi_table_view.hpp"
#include "leakflow/plot/score_view.hpp"
#include "leakflow/plot/table_view.hpp"
#include "leakflow/plugins/crypto_plot/poi_table_plot.hpp"
#include "leakflow/plugins/crypto_plot/score_plot.hpp"
#include "leakflow/plugins/crypto_plot/score_table_plot.hpp"
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
                    ScoreTablePlot::descriptor(),
                    PoiTablePlot::descriptor(),
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

    // One view per plot type, shared by every element of that type in this run, each
    // registered with the PlotRuntime so the UI draws/clears it alongside the traces.
    auto score_view = std::make_shared<leakflow::plot::ScoreView>();
    runtime->add_view(score_view);
    auto table_view = std::make_shared<leakflow::plot::TableView>();
    runtime->add_view(table_view);
    auto poi_table_view = std::make_shared<leakflow::plot::PoiTableView>();
    runtime->add_view(poi_table_view);

    registry.register_plugin(
        plugin_descriptors().front(),
        {
            {"ScorePlot", [view = std::move(score_view)](std::string name) {
                 return std::make_shared<ScorePlot>(view, std::move(name));
             }},
            {"ScoreTablePlot", [view = std::move(table_view)](std::string name) {
                 return std::make_shared<ScoreTablePlot>(view, std::move(name));
             }},
            {"PoiTablePlot", [view = std::move(poi_table_view)](std::string name) {
                 return std::make_shared<PoiTablePlot>(view, std::move(name));
             }},
        });
}

} // namespace leakflow::plugins::crypto_plot
