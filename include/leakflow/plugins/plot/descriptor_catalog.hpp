#pragma once

#include "leakflow/core/descriptor.hpp"
#include "leakflow/core/descriptor_registry.hpp"
#include "leakflow/core/element_factory_registry.hpp"
#include "leakflow/plot/plot_runtime.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace leakflow::plugins::plot {

inline constexpr auto plugin_name = "leakflow_plugins_plot";

[[nodiscard]] std::vector<PluginDescriptor> plugin_descriptors();
[[nodiscard]] const PluginDescriptor* find_plugin_descriptor(std::string_view name);
void register_plugin_descriptors(DescriptorRegistry& registry);
void register_element_factories(
    ElementFactoryRegistry& registry,
    std::shared_ptr<leakflow::plot::PlotRuntime> runtime);

} // namespace leakflow::plugins::plot
