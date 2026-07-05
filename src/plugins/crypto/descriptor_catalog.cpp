#include "leakflow/plugins/crypto/descriptor_catalog.hpp"

#include "leakflow/plugins/crypto/aes_leakage.hpp"
#include "leakflow/plugins/crypto/aes_leakage_hypothesis.hpp"
#include "leakflow/plugins/crypto/correlation_poi_to_plot_annotations.hpp"
#include "leakflow/plugins/crypto/cpa_attack.hpp"
#include "leakflow/plugins/crypto/dpa_attack.hpp"
#include "leakflow/plugins/crypto/attack_stats.hpp"
#include "leakflow/plugins/crypto/attack_stats_to_plot_annotations.hpp"
#include "leakflow/plugins/crypto/pearson_correlator.hpp"
#include "leakflow/plugins/crypto/poi_select.hpp"
#include "crypto_plugin_constants.hpp"

#include <memory>
#include <utility>

namespace leakflow::plugins::crypto {

std::vector<PluginDescriptor> plugin_descriptors()
{
    return {
        with_common_element_properties(PluginDescriptor{
            .name = plugin_name,
            .owner = crypto_author,
            .author = crypto_author,
            .license = crypto_license,
            .purpose = "shared library with crypto and SCA pipeline elements",
            .keywords = {"crypto", "sca", "aes", "leakage"},
            .elements = {
                AesLeakage::descriptor(),
                AesLeakageHypothesis::descriptor(),
                CpaAttack::descriptor(),
                DpaAttack::descriptor(),
                AttackStats::descriptor(),
                AttackStatsToPlotAnnotations::descriptor(),
                PearsonCorrelator::descriptor(),
                PoiSelect::descriptor(),
                CorrelationPoiToPlotAnnotations::descriptor(),
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
            {"AesLeakage", [](std::string name) {
                 return std::make_shared<AesLeakage>(std::move(name));
             }},
            {"AesLeakageHypothesis", [](std::string name) {
                 return std::make_shared<AesLeakageHypothesis>(std::move(name));
             }},
            {"CpaAttack", [](std::string name) {
                 return std::make_shared<CpaAttack>(std::move(name));
             }},
            {"DpaAttack", [](std::string name) {
                 return std::make_shared<DpaAttack>(std::move(name));
             }},
            {"AttackStats", [](std::string name) {
                 return std::make_shared<AttackStats>(std::move(name));
             }},
            {"AttackStatsToPlotAnnotations", [](std::string name) {
                 return std::make_shared<AttackStatsToPlotAnnotations>(std::move(name));
             }},
            {"PearsonCorrelator", [](std::string name) {
                 return std::make_shared<PearsonCorrelator>(std::move(name));
             }},
            {"PoiSelect", [](std::string name) {
                 return std::make_shared<PoiSelect>(std::move(name));
             }},
            {"CorrelationPoiToPlotAnnotations", [](std::string name) {
                 return std::make_shared<CorrelationPoiToPlotAnnotations>(std::move(name));
             }},
        });
}

} // namespace leakflow::plugins::crypto
