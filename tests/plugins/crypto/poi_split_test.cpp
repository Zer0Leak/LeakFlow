#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/base/statistics.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/summary_document.hpp"
#include "leakflow/plugins/crypto/crypto_elements.hpp"
#include "leakflow/plugins/crypto/descriptor_catalog.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace {

namespace crypto_plugin = leakflow::plugins::crypto;

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool expect_near(double actual, double expected, const char* message)
{
    if (std::abs(actual - expected) > 1.0e-6) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
        return false;
    }
    return true;
}

template <typename Function>
bool throws_invalid_argument(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

leakflow::Buffer torch_buffer(torch::Tensor tensor)
{
    auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    leakflow::Buffer buffer(payload.caps());
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    return buffer;
}

// Run the split analysis chain (PearsonCorrelator then PoiSelect) end to end. Callers
// that need to interrogate correlation state or accumulate keep their own correlator.
std::optional<leakflow::Buffer> select(crypto_plugin::PearsonCorrelator& corr, crypto_plugin::PoiSelect& poi,
                                       leakflow::Buffer features, leakflow::Buffer targets)
{
    leakflow::ElementInputs inputs;
    inputs.emplace("features", std::move(features));
    inputs.emplace("targets", std::move(targets));
    auto correlation = corr.process_inputs(std::move(inputs));
    if (!correlation) {
        return std::nullopt;
    }
    return poi.process(*correlation);
}

class TensorSource final : public leakflow::Element {
public:
    TensorSource(std::string name, torch::Tensor tensor)
        : Element(std::move(name))
        , tensor_(std::move(tensor))
    {
        set_element_identity("TensorSource", "Test/Source");
        add_output_pad(leakflow::Pad(
            "src", leakflow::PadDirection::Output, leakflow::Caps(leakflow::base::torch_tensor_caps_type)));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return torch_buffer(tensor_); }

private:
    torch::Tensor tensor_;
};

class LiveTensorSource final : public leakflow::Element {
public:
    explicit LiveTensorSource(std::string name)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "LiveTensorSource";
        descriptor.klass = "Source/Live/Test";
        descriptor.output_pads = {
            leakflow::Pad("src", leakflow::PadDirection::Output, leakflow::Caps(leakflow::base::torch_tensor_caps_type)),
        };
        descriptor.live_source = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
};

} // namespace

int main()
{
    const auto features = torch::tensor(
        {{1.0, 5.0, 1.0}, {2.0, 4.0, 1.0}, {3.0, 3.0, 1.0}, {4.0, 2.0, 1.0}, {5.0, 1.0, 1.0}},
        torch::TensorOptions().dtype(torch::kFloat32));
    const auto targets = torch::tensor(
        {{1.0, 5.0}, {2.0, 4.0}, {3.0, 3.0}, {4.0, 2.0}, {5.0, 1.0}},
        torch::TensorOptions().dtype(torch::kFloat32));

    // --- Correlator property/state contract (the stateful half) ---
    crypto_plugin::PearsonCorrelator corr;
    if (!expect(corr.property_as<std::string>("correlation_mode") == std::optional<std::string>("auto"),
            "PearsonCorrelator correlation_mode default was not auto")) {
        return 1;
    }
    if (!expect(corr.property_as<std::string>("active_correlation_mode") == std::optional<std::string>("recompute"),
            "PearsonCorrelator static active mode was not recompute")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&corr] {
            corr.set_property("active_correlation_mode", std::string("incremental"));
        }),
            "PearsonCorrelator active_correlation_mode was writable")) {
        return 1;
    }
    if (!expect(corr.can_replay(), "PearsonCorrelator static auto mode was not replay-safe")) {
        return 1;
    }

    // PoiSelect is stateless -> always replayable.
    crypto_plugin::PoiSelect poi;
    poi.set_property("top_k", leakflow::IntList{1, 1});
    poi.set_property("rank_by", leakflow::StringList{"abs"});
    if (!expect(poi.can_replay(), "PoiSelect was not replay-safe")) {
        return 1;
    }

    const auto output = select(corr, poi, torch_buffer(features), torch_buffer(targets));
    if (!expect(output.has_value(), "chain did not produce an output buffer")) {
        return 1;
    }
    if (!expect(output->caps().type() == crypto_plugin::correlation_poi_caps_type, "PoiSelect output caps type wrong")) {
        return 1;
    }
    if (!expect(output->metadata("payload.poi.method") == crypto_plugin::pearson_poi_method_id,
            "PoiSelect did not stamp method metadata")) {
        return 1;
    }
    if (!expect(output->metadata("payload.poi.features_count") == "3",
            "PoiSelect did not stamp searched feature count metadata")) {
        return 1;
    }
    if (!expect(output->metadata("payload.poi.correlation_mode") == "recompute",
            "static auto mode did not resolve to recompute")) {
        return 1;
    }
    if (!expect(output->metadata("payload.poi.observation_count") == "5",
            "PoiSelect did not carry recompute observation count metadata")) {
        return 1;
    }

    const auto payload = output->payload_as<crypto_plugin::CorrelationPoiPayload>();
    if (!expect(payload != nullptr, "PoiSelect payload type was wrong")) {
        return 1;
    }
    if (!expect(payload->result_count() == 2, "per-target result count was wrong")) {
        return 1;
    }
    if (!expect(payload->score_name() == "correlation", "per-target score name was wrong")) {
        return 1;
    }
    if (!expect(payload->result(0).unit == 0, "first generic target byte index was wrong")) {
        return 1;
    }
    if (!expect_near(payload->result(0).result[0][0][0].item<double>(), 0.0, "first target selected wrong feature")) {
        return 1;
    }
    if (!expect_near(payload->result(0).result[0][0][1].item<double>(), 1.0,
            "first target selected wrong correlation")) {
        return 1;
    }
    if (!expect_near(payload->result(1).result[0][0][1].item<double>(), -1.0,
            "second target selected wrong correlation")) {
        return 1;
    }

    // --- Correlation payload (the intermediate the split introduces) ---
    {
        crypto_plugin::PearsonCorrelator peek;
        leakflow::ElementInputs peek_inputs;
        peek_inputs.emplace("features", torch_buffer(features));
        peek_inputs.emplace("targets", torch_buffer(targets));
        const auto correlation_buffer = peek.process_inputs(std::move(peek_inputs));
        if (!expect(correlation_buffer && correlation_buffer->caps().type() == crypto_plugin::correlation_caps_type,
                "PearsonCorrelator output caps were wrong")) {
            return 1;
        }
        const auto correlation_payload = correlation_buffer->payload_as<crypto_plugin::CorrelationPayload>();
        if (!expect(correlation_payload != nullptr, "PearsonCorrelator payload type was wrong")) {
            return 1;
        }
        if (!expect(correlation_payload->feature_count() == 3 && correlation_payload->channel_count() == 1
                    && correlation_payload->byte_indexes().size() == 2,
                "CorrelationPayload layout was wrong")) {
            return 1;
        }
        if (!expect(correlation_buffer->metadata("payload.correlation.method")
                    == crypto_plugin::pearson_correlation_method_id,
                "PearsonCorrelator did not stamp method metadata")) {
            return 1;
        }
    }

    // --- Variable top_k on the selector ---
    {
        crypto_plugin::PearsonCorrelator vcorr;
        crypto_plugin::PoiSelect vpoi;
        vpoi.set_property("top_k", leakflow::IntList{2, 1});
        vpoi.set_property("rank_by", leakflow::StringList{"abs"});
        const auto out = select(vcorr, vpoi, torch_buffer(features), torch_buffer(targets));
        const auto p = out->payload_as<crypto_plugin::CorrelationPoiPayload>();
        if (!expect(p->result(0).result.size(1) == 2, "did not honor first target top_k")) {
            return 1;
        }
        if (!expect(p->result(1).result.size(1) == 1, "did not honor second target top_k")) {
            return 1;
        }
    }

    const auto streamed_features = torch::tensor(
        {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {5.0, 0.0}, {4.0, 0.0}, {3.0, 0.0}},
        torch::TensorOptions().dtype(torch::kFloat32));
    const auto streamed_targets = torch::tensor(
        {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}}, torch::TensorOptions().dtype(torch::kFloat32));

    // --- recompute mode uses only the current buffer ---
    {
        crypto_plugin::PearsonCorrelator rcorr;
        crypto_plugin::PoiSelect rpoi;
        rpoi.set_property("top_k", leakflow::IntList{1});
        (void)select(rcorr, rpoi, torch_buffer(streamed_features.slice(0, 0, 3)),
                     torch_buffer(streamed_targets.slice(0, 0, 3)));
        const auto out = select(rcorr, rpoi, torch_buffer(streamed_features.slice(0, 3, 6)),
                                torch_buffer(streamed_targets.slice(0, 3, 6)));
        const auto p = out->payload_as<crypto_plugin::CorrelationPoiPayload>();
        if (!expect_near(p->result(0).result[0][0][1].item<double>(), -1.0,
                "recompute mode did not use only the current buffer")) {
            return 1;
        }
        if (!expect(out->metadata("payload.poi.observation_count") == "3",
                "recompute mode accumulated its observation count")) {
            return 1;
        }
    }

    // --- auto mode goes incremental when live-driven ---
    {
        leakflow::Pipeline auto_live_pipeline;
        auto live_source = auto_live_pipeline.add(std::make_shared<LiveTensorSource>("live_features"));
        auto live_corr = std::make_shared<crypto_plugin::PearsonCorrelator>("auto_live_corr");
        auto live_corr_handle = auto_live_pipeline.add(live_corr);
        auto_live_pipeline.link(live_source, "src", live_corr_handle, "features");
        if (!expect(live_corr->is_live_driven(), "PearsonCorrelator did not receive propagated upstream liveness")) {
            return 1;
        }
        if (!expect(live_corr->property_as<std::string>("active_correlation_mode")
                    == std::optional<std::string>("incremental"),
                "live-driven active mode was not incremental")) {
            return 1;
        }
        if (!expect(!live_corr->can_replay(), "live-driven auto mode was marked replay-safe")) {
            return 1;
        }

        crypto_plugin::PoiSelect live_poi;
        live_poi.set_property("top_k", leakflow::IntList{1});
        leakflow::ElementInputs first;
        first.emplace("features", torch_buffer(streamed_features.slice(0, 0, 3)));
        first.emplace("targets", torch_buffer(streamed_targets.slice(0, 0, 3)));
        (void)live_corr->process_inputs(std::move(first));
        leakflow::ElementInputs second;
        second.emplace("features", torch_buffer(streamed_features.slice(0, 3, 6)));
        second.emplace("targets", torch_buffer(streamed_targets.slice(0, 3, 6)));
        const auto correlation = live_corr->process_inputs(std::move(second));
        const auto out = live_poi.process(*correlation);
        const auto p = out->payload_as<crypto_plugin::CorrelationPoiPayload>();
        const auto expected = leakflow::base::pearson_correlation(streamed_features, streamed_targets)[0][0].item<double>();
        if (!expect_near(p->result(0).result[0][0][1].item<double>(), expected,
                "live-driven auto mode did not incrementally merge multi-trace buffers")) {
            return 1;
        }
        if (!expect(out->metadata("payload.poi.correlation_mode") == "incremental",
                "live-driven auto mode did not resolve to incremental")) {
            return 1;
        }
        if (!expect(out->metadata("payload.poi.observation_count") == "6",
                "live-driven auto mode observation count was wrong")) {
            return 1;
        }
    }

    // --- explicit incremental accumulation + start() reset ---
    {
        crypto_plugin::PearsonCorrelator icorr;
        icorr.set_property("correlation_mode", std::string("incremental"));
        if (!expect(!icorr.can_replay(), "incremental mode was marked replay-safe")) {
            return 1;
        }
        icorr.set_property("correlation_mode", std::string("recompute"));
        if (!expect(icorr.can_replay(), "recompute mode was marked non-replayable")) {
            return 1;
        }
        icorr.set_property("correlation_mode", std::string("incremental"));

        crypto_plugin::PoiSelect ipoi;
        ipoi.set_property("top_k", leakflow::IntList{1});
        std::optional<leakflow::Buffer> out;
        for (std::int64_t row = 0; row < streamed_features.size(0); ++row) {
            out = select(icorr, ipoi, torch_buffer(streamed_features.slice(0, row, row + 1)),
                         torch_buffer(streamed_targets.slice(0, row, row + 1)));
        }
        const auto p = out->payload_as<crypto_plugin::CorrelationPoiPayload>();
        const auto expected = leakflow::base::pearson_correlation(streamed_features, streamed_targets)[0][0].item<double>();
        if (!expect_near(p->result(0).result[0][0][1].item<double>(), expected,
                "incremental mode did not accumulate one-row buffers")) {
            return 1;
        }
        if (!expect(out->metadata("payload.poi.observation_count") == "6",
                "incremental observation count was wrong")) {
            return 1;
        }

        icorr.start();
        const auto restarted = select(icorr, ipoi, torch_buffer(streamed_features.slice(0, 0, 1)),
                                      torch_buffer(streamed_targets.slice(0, 0, 1)));
        if (!expect(restarted->metadata("payload.poi.observation_count") == "1",
                "start did not reset incremental state")) {
            return 1;
        }

        if (!expect(throws_invalid_argument([&icorr] {
                icorr.set_property("correlation_mode", std::string("interactive"));
            }),
                "PearsonCorrelator accepted an invalid correlation_mode")) {
            return 1;
        }
    }

    // --- annotation converter preserves exact / rejects out-of-range norm values ---
    {
        auto exact = crypto_plugin::CorrelationPoiPayload(
            std::vector<crypto_plugin::CorrelationPoiResult>{crypto_plugin::CorrelationPoiResult{
                .unit = 0,
                .result = torch::tensor({{{7.0, 0.625}}}, torch::TensorOptions().dtype(torch::kFloat64)),
            }},
            "custom-score");
        leakflow::Buffer exact_buffer{leakflow::Caps(crypto_plugin::correlation_poi_caps_type)};
        exact_buffer.set_metadata("payload.poi.method", crypto_plugin::pearson_poi_method_id);
        exact_buffer.set_payload(std::make_shared<crypto_plugin::CorrelationPoiPayload>(std::move(exact)));
        crypto_plugin::CorrelationPoiToPlotAnnotations converter;
        const auto out = converter.process(exact_buffer);
        const auto ann = out->payload_as<leakflow::base::PlotAnnotationPayload>();
        if (!expect(ann->annotation(0).norm_value && *ann->annotation(0).norm_value == 0.625,
                "converter did not preserve the exact Pearson correlation norm_value")) {
            return 1;
        }

        auto bad = crypto_plugin::CorrelationPoiPayload(
            std::vector<crypto_plugin::CorrelationPoiResult>{crypto_plugin::CorrelationPoiResult{
                .unit = 0,
                .result = torch::tensor({{{7.0, 1.25}}}, torch::TensorOptions().dtype(torch::kFloat64)),
            }},
            "correlation");
        leakflow::Buffer bad_buffer{leakflow::Caps(crypto_plugin::correlation_poi_caps_type)};
        bad_buffer.set_metadata("payload.poi.method", crypto_plugin::pearson_poi_method_id);
        bad_buffer.set_payload(std::make_shared<crypto_plugin::CorrelationPoiPayload>(std::move(bad)));
        if (!expect(throws_invalid_argument([&converter, &bad_buffer] { (void)converter.process(bad_buffer); }),
                "converter clamped an out-of-range Pearson correlation norm_value")) {
            return 1;
        }
    }

    // --- broadcast targets flatten into groups ---
    {
        const auto broadcast_targets = targets.reshape({1, 5, 2}).repeat({2, 1, 1});
        crypto_plugin::PearsonCorrelator bcorr;
        crypto_plugin::PoiSelect bpoi;
        bpoi.set_property("top_k", leakflow::IntList{1});
        const auto out = select(bcorr, bpoi, torch_buffer(features), torch_buffer(broadcast_targets));
        const auto p = out->payload_as<crypto_plugin::CorrelationPoiPayload>();
        if (!expect(p->result_count() == 4, "did not flatten broadcast target groups")) {
            return 1;
        }
    }

    // --- AES metadata flows through corr -> poi, and into annotations ---
    const auto aes_targets = targets.reshape({1, 5, 2}).repeat({2, 1, 1});
    auto aes_target_buffer = torch_buffer(aes_targets);
    aes_target_buffer.set_metadata("payload.leakage.model", crypto_plugin::aes_leakage_model_id);
    aes_target_buffer.set_metadata("payload.leakage.byte_indexes", "[3,5]");
    aes_target_buffer.set_metadata("payload.leakage.channels", "HW(m),HW(y)");
    aes_target_buffer.set_metadata("payload.crypto.algorithm", "AES");

    crypto_plugin::PearsonCorrelator aes_corr;
    crypto_plugin::PoiSelect aes_poi;
    aes_poi.set_property("top_k", leakflow::IntList{1});
    aes_poi.set_property("rank_by", leakflow::StringList{"abs", "abs"});
    const auto aes_output = select(aes_corr, aes_poi, torch_buffer(features), aes_target_buffer);
    if (!expect(aes_output->metadata("payload.leakage.byte_indexes") == "[3,5]",
            "did not forward AES byte-index metadata")) {
        return 1;
    }
    if (!expect(aes_output->metadata("payload.leakage.channels") == "HW(m),HW(y)",
            "did not forward AES channel metadata")) {
        return 1;
    }
    if (!expect(aes_output->metadata("payload.poi.features_count") == "3",
            "did not stamp searched feature count metadata")) {
        return 1;
    }
    const auto aes_payload = aes_output->payload_as<crypto_plugin::CorrelationPoiPayload>();
    if (!expect(aes_payload != nullptr && aes_payload->result_count() == 2,
            "AES byte-group count was wrong")) {
        return 1;
    }
    if (!expect(aes_payload->result(0).unit == 3 && aes_payload->result(1).unit == 5,
            "AES byte indexes were wrong")) {
        return 1;
    }
    if (!expect(aes_payload->result(0).result.dim() == 3 && aes_payload->result(0).result.size(0) == 2
                && aes_payload->result(0).result.size(1) == 1 && aes_payload->result(0).result.size(2) == 2,
            "AES result tensor shape was wrong")) {
        return 1;
    }

    leakflow::SummarySection aes_payload_summary("Payload");
    aes_payload->describe(aes_payload_summary, 1);
    bool saw_byte_3 = false;
    bool saw_byte_5 = false;
    for (const auto& field : aes_payload_summary.fields) {
        if (field.label == "result" && field.value.text == "(unit: 3, shape: [2, 1, 2])") {
            saw_byte_3 = true;
        }
        if (field.label == "result" && field.value.text == "(unit: 5, shape: [2, 1, 2])") {
            saw_byte_5 = true;
        }
    }
    if (!expect(saw_byte_3 && saw_byte_5, "CorrelationPoiPayload summary did not include per-byte result shapes")) {
        return 1;
    }

    crypto_plugin::CorrelationPoiToPlotAnnotations annotation_converter;
    annotation_converter.set_property("precision", std::int64_t{2});
    const auto annotation_output = annotation_converter.process(aes_output);
    if (!expect(annotation_output.has_value() && annotation_output->caps().type()
                == leakflow::base::plot_annotation_caps_type,
            "annotation converter emitted wrong caps type")) {
        return 1;
    }
    const auto annotations = annotation_output->payload_as<leakflow::base::PlotAnnotationPayload>();
    if (!expect(annotations != nullptr && annotations->annotation_count() == 4,
            "annotation count was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).label == "unit_3.HW(m)", "did not use target label metadata")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).norm_value && *annotations->annotation(0).norm_value == 1.0,
            "annotation norm_value was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(1).norm_value && *annotations->annotation(1).norm_value == -1.0,
            "negative correlation norm_value was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).fields.size() == 4
                && annotations->annotation(0).fields[3].first == "correlation"
                && annotations->annotation(0).fields[3].second == "1.00",
            "annotation score field was wrong")) {
        return 1;
    }

    // --- linked pipeline: sources -> corr -> poi ---
    {
        leakflow::Pipeline pipeline;
        auto feature_source = pipeline.add(std::make_shared<TensorSource>("features_src", features));
        auto target_source = pipeline.add(std::make_shared<TensorSource>("targets_src", targets));
        auto correlator = pipeline.add(std::make_shared<crypto_plugin::PearsonCorrelator>("corr"));
        auto selector_element = std::make_shared<crypto_plugin::PoiSelect>("poi");
        selector_element->set_property("top_k", leakflow::IntList{1});
        auto selector = pipeline.add(std::move(selector_element));
        pipeline.link(feature_source, "src", correlator, "features");
        pipeline.link(target_source, "src", correlator, "targets");
        pipeline.link(correlator, "correlation", selector, "correlation");
        const auto pipeline_output = pipeline.run();
        if (!expect(pipeline_output.has_value(), "linked corr->poi pipeline produced no output")) {
            return 1;
        }
        if (!expect(pipeline_output->payload_as<crypto_plugin::CorrelationPoiPayload>() != nullptr,
                "linked corr->poi pipeline produced wrong payload")) {
            return 1;
        }
    }

    // --- error paths ---
    if (!expect(throws_invalid_argument([&features] {
            crypto_plugin::PearsonCorrelator bad;
            leakflow::ElementInputs bad_inputs;
            bad_inputs.emplace("features", torch_buffer(features));
            (void)bad.process_inputs(std::move(bad_inputs));
        }),
            "PearsonCorrelator accepted missing targets input")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&features, &targets] {
            crypto_plugin::PearsonCorrelator bad_corr;
            crypto_plugin::PoiSelect bad_poi;
            bad_poi.set_property("top_k", leakflow::IntList{100});
            (void)select(bad_corr, bad_poi, torch_buffer(features), torch_buffer(targets));
        }),
            "PoiSelect accepted top_k larger than feature count")) {
        return 1;
    }

    // --- descriptor catalog reflects the split ---
    const auto descriptors = crypto_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "crypto plugin descriptor count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 12, "crypto plugin element count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[6].type_name == "PearsonCorrelator"
                && descriptors[0].elements[6].output_pads[0].caps().type() == crypto_plugin::correlation_caps_type,
            "PearsonCorrelator descriptor was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[7].type_name == "PoiSelect"
                && descriptors[0].elements[7].output_pads[0].caps().type() == crypto_plugin::correlation_poi_caps_type,
            "PoiSelect descriptor was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[8].type_name == "CorrelationPoiToPlotAnnotations",
            "CorrelationPoiToPlotAnnotations descriptor type name was wrong")) {
        return 1;
    }

    return 0;
}
