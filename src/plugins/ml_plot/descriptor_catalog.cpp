#include "leakflow/plugins/ml_plot/descriptor_catalog.hpp"

#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plot/table_view.hpp"
#include "leakflow/plugins/ml_plot/clustering_metrics_table_plot.hpp"
#include "ml_plot_plugin_constants.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace leakflow::plugins::ml_plot {

std::vector<PluginDescriptor> plugin_descriptors() {
  return {
      with_common_element_properties(PluginDescriptor{
          .name = plugin_name,
          .owner = ml_plot_author,
          .author = ml_plot_author,
          .license = ml_plot_license,
          .version = ml_plot_version,
          .purpose = "bridge structured ML evaluation results to domain-free "
                     "interactive plots",
          .keywords = {"ml", "clustering", "evaluation", "metrics", "plot",
                       "table"},
          .elements =
              {
                  ClusteringMetricsTablePlot::descriptor(),
              },
      }),
  };
}

const PluginDescriptor *find_plugin_descriptor(std::string_view name) {
  static const auto descriptors = plugin_descriptors();
  for (const auto &descriptor : descriptors) {
    if (descriptor.name == name) {
      return &descriptor;
    }
  }
  return nullptr;
}

void register_plugin_descriptors(DescriptorRegistry &registry) {
  registry.register_plugins(plugin_descriptors());
}

void register_element_factories(
    ElementFactoryRegistry &registry,
    std::shared_ptr<leakflow::plot::PlotRuntime> runtime) {
  if (!runtime) {
    throw std::invalid_argument("ClusteringMetricsTablePlot factory "
                                "registration requires a PlotRuntime");
  }

  auto table_view = std::make_shared<leakflow::plot::TableView>();
  runtime->add_view(table_view);
  registry.register_plugin(
      plugin_descriptors().front(),
      {
          {"ClusteringMetricsTablePlot",
           [view = std::move(table_view)](std::string name) {
             return std::make_shared<ClusteringMetricsTablePlot>(
                 view, std::move(name));
           }},
      });
}

} // namespace leakflow::plugins::ml_plot
