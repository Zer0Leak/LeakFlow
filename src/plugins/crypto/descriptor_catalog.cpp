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
#include "leakflow/plugins/crypto/pearson_correlator.hpp"
#include "leakflow/plugins/crypto/poi_select.hpp"
#include "crypto_plugin_constants.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
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

namespace {

void write_blob(const std::filesystem::path& file, const std::vector<char>& data)
{
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("payload codec could not open for writing: " + file.string());
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) {
        throw std::runtime_error("payload codec failed while writing: " + file.string());
    }
}

[[nodiscard]] std::vector<char> read_blob(const std::filesystem::path& file)
{
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("payload codec could not open for reading: " + file.string());
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

[[nodiscard]] torch::Tensor byte_indexes_to_tensor(const std::vector<std::uint16_t>& byte_indexes)
{
    auto tensor = torch::empty({static_cast<std::int64_t>(byte_indexes.size())}, torch::kInt64);
    auto accessor = tensor.accessor<std::int64_t, 1>();
    for (std::size_t index = 0; index < byte_indexes.size(); ++index) {
        accessor[static_cast<std::int64_t>(index)] = byte_indexes[index];
    }
    return tensor;
}

[[nodiscard]] std::vector<std::uint16_t> byte_indexes_from_tensor(const torch::Tensor& tensor)
{
    const auto contiguous = tensor.to(torch::kInt64).contiguous();
    const auto accessor = contiguous.accessor<std::int64_t, 1>();
    std::vector<std::uint16_t> byte_indexes;
    byte_indexes.reserve(static_cast<std::size_t>(contiguous.size(0)));
    for (std::int64_t index = 0; index < contiguous.size(0); ++index) {
        byte_indexes.push_back(static_cast<std::uint16_t>(accessor[index]));
    }
    return byte_indexes;
}

} // namespace

void register_payload_codecs(PayloadCodecRegistry& codecs)
{
    // CorrelationPayload: [grouped, byte_indexes, channel_count, feature_count,
    // score_name, observation_count] pickled as one IValue tuple.
    codecs.register_codec(
        correlation_caps_type,
        PayloadCodec{
            .save =
                [](const Payload& payload, const std::filesystem::path& dir) {
                    const auto* correlation = dynamic_cast<const CorrelationPayload*>(&payload);
                    if (correlation == nullptr) {
                        throw std::invalid_argument("correlation codec: payload is not a CorrelationPayload");
                    }
                    auto tuple = c10::ivalue::Tuple::create({
                        correlation->grouped_correlation(),
                        byte_indexes_to_tensor(correlation->byte_indexes()),
                        correlation->channel_count(),
                        correlation->feature_count(),
                        correlation->score_name(),
                        correlation->observation_count(),
                    });
                    write_blob(dir / "payload.pt", torch::pickle_save(c10::IValue(std::move(tuple))));
                },
            .load =
                [](const std::filesystem::path& dir) -> std::shared_ptr<Payload> {
                    const auto value = torch::pickle_load(read_blob(dir / "payload.pt"));
                    const auto& elements = value.toTupleRef().elements();
                    return std::make_shared<CorrelationPayload>(
                        elements.at(0).toTensor(),
                        byte_indexes_from_tensor(elements.at(1).toTensor()),
                        elements.at(2).toInt(),
                        elements.at(3).toInt(),
                        elements.at(4).toStringRef(),
                        elements.at(5).toInt());
                },
        });

    // CorrelationPoiPayload: [byte_indexes, list<result tensor>, score_name].
    codecs.register_codec(
        correlation_poi_caps_type,
        PayloadCodec{
            .save =
                [](const Payload& payload, const std::filesystem::path& dir) {
                    const auto* poi = dynamic_cast<const CorrelationPoiPayload*>(&payload);
                    if (poi == nullptr) {
                        throw std::invalid_argument("correlation-poi codec: payload is not a CorrelationPoiPayload");
                    }
                    std::vector<std::uint16_t> byte_indexes;
                    c10::List<torch::Tensor> result_tensors;
                    byte_indexes.reserve(poi->result_count());
                    for (const auto& result : poi->results()) {
                        byte_indexes.push_back(result.target_byte_index);
                        result_tensors.push_back(result.result);
                    }
                    auto tuple = c10::ivalue::Tuple::create({
                        byte_indexes_to_tensor(byte_indexes),
                        result_tensors,
                        poi->score_name(),
                    });
                    write_blob(dir / "payload.pt", torch::pickle_save(c10::IValue(std::move(tuple))));
                },
            .load =
                [](const std::filesystem::path& dir) -> std::shared_ptr<Payload> {
                    const auto value = torch::pickle_load(read_blob(dir / "payload.pt"));
                    const auto& elements = value.toTupleRef().elements();
                    const auto byte_indexes = byte_indexes_from_tensor(elements.at(0).toTensor());
                    const auto result_tensors = elements.at(1).toTensorList();
                    std::vector<CorrelationPoiResult> results;
                    results.reserve(byte_indexes.size());
                    for (std::size_t index = 0; index < byte_indexes.size(); ++index) {
                        results.push_back(CorrelationPoiResult{
                            .target_byte_index = byte_indexes[index],
                            .result = result_tensors.get(index),
                        });
                    }
                    return std::make_shared<CorrelationPoiPayload>(std::move(results), elements.at(2).toStringRef());
                },
        });
}

} // namespace leakflow::plugins::crypto
