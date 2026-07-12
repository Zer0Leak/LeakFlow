#include "leakflow/plugins/crypto/descriptor_catalog.hpp"

#include "leakflow/plugins/crypto/aes_leakage.hpp"
#include "leakflow/plugins/crypto/aes_leakage_hypothesis.hpp"
#include "leakflow/plugins/crypto/correlation_poi_to_plot_annotations.hpp"
#include "leakflow/plugins/crypto/cpa_attack.hpp"
#include "leakflow/plugins/crypto/dpa_attack.hpp"
#include "leakflow/plugins/crypto/attack_stats.hpp"
#include "leakflow/plugins/crypto/attack_stats_to_plot_annotations.hpp"
#include "leakflow/plugins/crypto/correlation_payload.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"
#include "leakflow/plugins/crypto/correlation_poi_to_indexes.hpp"
#include "leakflow/plugins/crypto/hw_class.hpp"
#include "leakflow/plugins/crypto/poi_correlation.hpp"
#include "leakflow/plugins/crypto/pearson_correlator.hpp"
#include "leakflow/plugins/crypto/poi_select.hpp"
#include "crypto_plugin_constants.hpp"

#include "leakflow/base/buffer_archive.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <vector>

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
                CorrelationPoiToIndexes::descriptor(),
                HwClass::descriptor(),
                PoiCorrelation::descriptor(),
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
            {"CorrelationPoiToIndexes", [](std::string name) {
                 return std::make_shared<CorrelationPoiToIndexes>(std::move(name));
             }},
            {"HwClass", [](std::string name) {
                 return std::make_shared<HwClass>(std::move(name));
             }},
            {"PoiCorrelation", [](std::string name) {
                 return std::make_shared<PoiCorrelation>(std::move(name));
             }},
        });
}

namespace {

[[nodiscard]] torch::Tensor unit_indexes_to_tensor(const std::vector<std::int64_t>& unit_indexes)
{
    auto tensor = torch::empty({static_cast<std::int64_t>(unit_indexes.size())}, torch::kInt64);
    auto accessor = tensor.accessor<std::int64_t, 1>();
    for (std::size_t index = 0; index < unit_indexes.size(); ++index) {
        accessor[static_cast<std::int64_t>(index)] = unit_indexes[index];
    }
    return tensor;
}

[[nodiscard]] std::vector<std::int64_t> unit_indexes_from_tensor(const torch::Tensor& tensor)
{
    const auto contiguous = tensor.to(torch::kInt64).contiguous();
    const auto accessor = contiguous.accessor<std::int64_t, 1>();
    std::vector<std::int64_t> unit_indexes;
    unit_indexes.reserve(static_cast<std::size_t>(contiguous.size(0)));
    for (std::int64_t index = 0; index < contiguous.size(0); ++index) {
        unit_indexes.push_back(accessor[index]);
    }
    return unit_indexes;
}

} // namespace

void register_payload_codecs(PayloadCodecRegistry& codecs)
{
    // CorrelationPayload: [grouped, unit_indexes, channel_count, feature_count,
    // score_name, observation_count] pickled as one IValue tuple.
    codecs.register_codec(
        correlation_caps_type,
        PayloadCodec{
            .save =
                [](const Payload& payload, leakflow::base::BufferArchiveWriter& archive) {
                    const auto* correlation = dynamic_cast<const CorrelationPayload*>(&payload);
                    if (correlation == nullptr) {
                        throw std::invalid_argument("correlation codec: payload is not a CorrelationPayload");
                    }
                    archive.write_tensor("grouped_correlation", correlation->grouped_correlation());
                    archive.write_tensor("unit_indexes", unit_indexes_to_tensor(correlation->unit_indexes()));
                    archive.write_int("channel_count", correlation->channel_count());
                    archive.write_int("feature_count", correlation->feature_count());
                    archive.write_string("score_name", correlation->score_name());
                    archive.write_int("observation_count", correlation->observation_count());
                },
            .load =
                [](const leakflow::base::BufferArchiveReader& archive) -> std::shared_ptr<Payload> {
                    return std::make_shared<CorrelationPayload>(
                        archive.read_tensor("grouped_correlation"),
                        unit_indexes_from_tensor(archive.read_tensor("unit_indexes")),
                        archive.read_int("channel_count"),
                        archive.read_int("feature_count"),
                        archive.read_string("score_name"),
                        archive.read_int("observation_count"));
                },
        });

    // CorrelationPoiPayload: [unit_indexes, list<result tensor>, score_name].
    codecs.register_codec(
        correlation_poi_caps_type,
        PayloadCodec{
            .save =
                [](const Payload& payload, leakflow::base::BufferArchiveWriter& archive) {
                    const auto* poi = dynamic_cast<const CorrelationPoiPayload*>(&payload);
                    if (poi == nullptr) {
                        throw std::invalid_argument("correlation-poi codec: payload is not a CorrelationPoiPayload");
                    }
                    std::vector<std::int64_t> unit_indexes;
                    unit_indexes.reserve(poi->unit_count());
                    const auto& results = poi->results();
                    for (std::size_t index = 0; index < results.size(); ++index) {
                        unit_indexes.push_back(results[index].unit_index);
                        archive.write_tensor("result_" + std::to_string(index), results[index].result);
                    }
                    archive.write_tensor("unit_indexes", unit_indexes_to_tensor(unit_indexes));
                    archive.write_int("unit_count", static_cast<std::int64_t>(results.size()));
                    archive.write_string("score_name", poi->score_name());
                },
            .load =
                [](const leakflow::base::BufferArchiveReader& archive) -> std::shared_ptr<Payload> {
                    const auto unit_indexes = unit_indexes_from_tensor(archive.read_tensor("unit_indexes"));
                    const auto unit_count = static_cast<std::size_t>(archive.read_int("unit_count"));
                    std::vector<CorrelationPoiResult> results;
                    results.reserve(unit_count);
                    for (std::size_t index = 0; index < unit_count; ++index) {
                        results.push_back(CorrelationPoiResult{
                            .unit_index = unit_indexes.at(index),
                            .result = archive.read_tensor("result_" + std::to_string(index)),
                        });
                    }
                    return std::make_shared<CorrelationPoiPayload>(std::move(results), archive.read_string("score_name"));
                },
        });
}

} // namespace leakflow::plugins::crypto
