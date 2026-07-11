#include "leakflow/plugins/ml/descriptor_catalog.hpp"

#include "leakflow/plugins/ml/clustering_stats.hpp"
#include "leakflow/plugins/ml/feature_select.hpp"
#include "leakflow/plugins/ml/gaussian_mixture_element.hpp"
#include "ml_plugin_constants.hpp"

#include <memory>
#include <utility>

namespace leakflow::plugins::ml {

std::vector<PluginDescriptor> plugin_descriptors()
{
    return {
        with_common_element_properties(PluginDescriptor{
            .name = plugin_name,
            .owner = ml_author,
            .author = ml_author,
            .license = ml_license,
            .version = ml_version,
            .purpose = "generic machine-learning / statistics pipeline elements (clustering, evaluation)",
            .keywords = {"ml", "clustering", "gmm", "statistics"},
            .elements = {
                FeatureSelect::descriptor(),
                GaussianMixtureElement::descriptor(),
                ClusteringStats::descriptor(),
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
            {"FeatureSelect", [](std::string name) {
                 return std::make_shared<FeatureSelect>(std::move(name));
             }},
            {"GaussianMixture", [](std::string name) {
                 return std::make_shared<GaussianMixtureElement>(std::move(name));
             }},
            {"ClusteringStats", [](std::string name) {
                 return std::make_shared<ClusteringStats>(std::move(name));
             }},
        });
}

} // namespace leakflow::plugins::ml
