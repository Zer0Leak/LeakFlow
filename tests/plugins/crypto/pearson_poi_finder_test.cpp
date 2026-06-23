#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/plot_annotation_payload.hpp"
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

class TensorSource final : public leakflow::Element {
public:
    TensorSource(std::string name, torch::Tensor tensor)
        : Element(std::move(name))
        , tensor_(std::move(tensor))
    {
        set_element_identity("TensorSource", "Test/Source");
        add_output_pad(leakflow::Pad(
            "src",
            leakflow::PadDirection::Output,
            leakflow::Caps(leakflow::base::torch_tensor_caps_type)));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        return torch_buffer(tensor_);
    }

private:
    torch::Tensor tensor_;
};

} // namespace

int main()
{
    namespace crypto_plugin = leakflow::plugins::crypto;

    const auto features = torch::tensor(
        {
            {1.0, 5.0, 1.0},
            {2.0, 4.0, 1.0},
            {3.0, 3.0, 1.0},
            {4.0, 2.0, 1.0},
            {5.0, 1.0, 1.0},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    const auto targets = torch::tensor(
        {
            {1.0, 5.0},
            {2.0, 4.0},
            {3.0, 3.0},
            {4.0, 2.0},
            {5.0, 1.0},
        },
        torch::TensorOptions().dtype(torch::kFloat32));

    crypto_plugin::PearsonPoiFinder finder;
    finder.set_property("top_k", leakflow::IntList{1, 1});
    // rank_by is per leakage feature/channel; this generic target has one channel.
    finder.set_property("rank_by", leakflow::StringList{"abs"});

    leakflow::ElementInputs inputs;
    inputs.emplace("features", torch_buffer(features));
    inputs.emplace("targets", torch_buffer(targets));
    const auto output = finder.process_inputs(std::move(inputs));
    if (!expect(output.has_value(), "PearsonPoiFinder did not produce an output buffer")) {
        return 1;
    }
    if (!expect(output->caps().type() == crypto_plugin::correlation_poi_caps_type,
            "PearsonPoiFinder output caps type was wrong")) {
        return 1;
    }
    if (!expect(output->metadata("payload.poi.method") == crypto_plugin::pearson_poi_method_id,
            "PearsonPoiFinder did not stamp method metadata")) {
        return 1;
    }
    if (!expect(output->metadata("payload.poi.features_count") == "3",
            "PearsonPoiFinder did not stamp searched feature count metadata")) {
        return 1;
    }
    if (!expect(!output->has_metadata("payload.poi.result_groups"),
            "PearsonPoiFinder still stamped redundant result group metadata")) {
        return 1;
    }
    if (!expect(!output->has_metadata("payload.poi.score"),
            "PearsonPoiFinder still stamped redundant score metadata")) {
        return 1;
    }

    const auto payload = output->payload_as<crypto_plugin::CorrelationPoiPayload>();
    if (!expect(payload != nullptr, "PearsonPoiFinder payload type was wrong")) {
        return 1;
    }
    if (!expect(payload->type_name() == crypto_plugin::correlation_poi_caps_type,
            "CorrelationPoiPayload type name was wrong")) {
        return 1;
    }
    if (!expect(payload->result_count() == 2, "PearsonPoiFinder per-target result count was wrong")) {
        return 1;
    }
    if (!expect(payload->score_name() == "correlation", "PearsonPoiFinder per-target score name was wrong")) {
        return 1;
    }
    if (!expect(payload->result(0).target_byte_index == 0,
            "PearsonPoiFinder first generic target byte index was wrong")) {
        return 1;
    }
    if (!expect(payload->result(0).result.dim() == 3 && payload->result(0).result.size(0) == 1
                && payload->result(0).result.size(1) == 1 && payload->result(0).result.size(2) == 2,
            "PearsonPoiFinder first target pair shape was wrong")) {
        return 1;
    }
    if (!expect_near(payload->result(0).result[0][0][0].item<double>(), 0.0,
            "PearsonPoiFinder first target selected wrong feature")) {
        return 1;
    }
    if (!expect_near(payload->result(0).result[0][0][1].item<double>(), 1.0,
            "PearsonPoiFinder first target selected wrong correlation")) {
        return 1;
    }
    if (!expect_near(payload->result(1).result[0][0][0].item<double>(), 0.0,
            "PearsonPoiFinder second target selected wrong feature")) {
        return 1;
    }
    if (!expect_near(payload->result(1).result[0][0][1].item<double>(), -1.0,
            "PearsonPoiFinder second target selected wrong correlation")) {
        return 1;
    }

    crypto_plugin::PearsonPoiFinder variable_top_k_finder;
    variable_top_k_finder.set_property("top_k", leakflow::IntList{2, 1});
    variable_top_k_finder.set_property("rank_by", leakflow::StringList{"abs"});
    leakflow::ElementInputs variable_inputs;
    variable_inputs.emplace("features", torch_buffer(features));
    variable_inputs.emplace("targets", torch_buffer(targets));
    const auto variable_output = variable_top_k_finder.process_inputs(std::move(variable_inputs));
    const auto variable_payload = variable_output->payload_as<crypto_plugin::CorrelationPoiPayload>();
    if (!expect(variable_payload->result(0).result.size(1) == 2,
            "PearsonPoiFinder did not honor first target top_k")) {
        return 1;
    }
    if (!expect(variable_payload->result(1).result.size(1) == 1,
            "PearsonPoiFinder did not honor second target top_k")) {
        return 1;
    }

    auto exact_correlation_payload = crypto_plugin::CorrelationPoiPayload(
        std::vector<crypto_plugin::CorrelationPoiResult>{
            crypto_plugin::CorrelationPoiResult{
                .target_byte_index = 0,
                .result = torch::tensor({{{7.0, 0.625}}}, torch::TensorOptions().dtype(torch::kFloat64)),
            },
        },
        "custom-score");
    leakflow::Buffer exact_correlation_buffer{leakflow::Caps(crypto_plugin::correlation_poi_caps_type)};
    exact_correlation_buffer.set_metadata("payload.poi.method", crypto_plugin::pearson_poi_method_id);
    exact_correlation_buffer.set_payload(
        std::make_shared<crypto_plugin::CorrelationPoiPayload>(std::move(exact_correlation_payload)));

    crypto_plugin::CorrelationPoiToPlotAnnotations exact_correlation_converter;
    const auto exact_correlation_output = exact_correlation_converter.process(exact_correlation_buffer);
    const auto exact_correlation_annotations =
        exact_correlation_output->payload_as<leakflow::base::PlotAnnotationPayload>();
    if (!expect(exact_correlation_annotations->annotation(0).norm_value
                && *exact_correlation_annotations->annotation(0).norm_value == 0.625,
            "CorrelationPoiToPlotAnnotations did not preserve the exact Pearson correlation norm_value")) {
        return 1;
    }

    auto out_of_range_correlation_payload = crypto_plugin::CorrelationPoiPayload(
        std::vector<crypto_plugin::CorrelationPoiResult>{
            crypto_plugin::CorrelationPoiResult{
                .target_byte_index = 0,
                .result = torch::tensor({{{7.0, 1.25}}}, torch::TensorOptions().dtype(torch::kFloat64)),
            },
        },
        "correlation");
    leakflow::Buffer out_of_range_correlation_buffer{leakflow::Caps(crypto_plugin::correlation_poi_caps_type)};
    out_of_range_correlation_buffer.set_metadata("payload.poi.method", crypto_plugin::pearson_poi_method_id);
    out_of_range_correlation_buffer.set_payload(
        std::make_shared<crypto_plugin::CorrelationPoiPayload>(std::move(out_of_range_correlation_payload)));
    if (!expect(throws_invalid_argument([&exact_correlation_converter, &out_of_range_correlation_buffer] {
            (void)exact_correlation_converter.process(out_of_range_correlation_buffer);
        }),
            "CorrelationPoiToPlotAnnotations clamped an out-of-range Pearson correlation norm_value")) {
        return 1;
    }

    const auto broadcast_targets = targets.reshape({1, 5, 2}).repeat({2, 1, 1});
    crypto_plugin::PearsonPoiFinder broadcast_finder;
    broadcast_finder.set_property("top_k", leakflow::IntList{1});
    leakflow::ElementInputs broadcast_inputs;
    broadcast_inputs.emplace("features", torch_buffer(features));
    broadcast_inputs.emplace("targets", torch_buffer(broadcast_targets));
    const auto broadcast_output = broadcast_finder.process_inputs(std::move(broadcast_inputs));
    const auto broadcast_payload = broadcast_output->payload_as<crypto_plugin::CorrelationPoiPayload>();
    if (!expect(broadcast_payload->result_count() == 4,
            "PearsonPoiFinder did not flatten broadcast target groups")) {
        return 1;
    }

    const auto aes_targets = targets.reshape({1, 5, 2}).repeat({2, 1, 1});
    auto aes_target_buffer = torch_buffer(aes_targets);
    aes_target_buffer.set_metadata("payload.leakage.model", crypto_plugin::aes_leakage_model_id);
    aes_target_buffer.set_metadata("payload.leakage.byte_indexes", "[3,5]");
    aes_target_buffer.set_metadata("payload.leakage.channels", "HW(m),HW(y)");
    aes_target_buffer.set_metadata("payload.crypto.algorithm", "AES");

    crypto_plugin::PearsonPoiFinder aes_finder;
    aes_finder.set_property("top_k", leakflow::IntList{1});
    // rank_by length must match the number of leakage features/channels (HW(m), HW(y)).
    aes_finder.set_property("rank_by", leakflow::StringList{"abs", "abs"});
    leakflow::ElementInputs aes_inputs;
    aes_inputs.emplace("features", torch_buffer(features));
    aes_inputs.emplace("targets", aes_target_buffer);
    const auto aes_output = aes_finder.process_inputs(std::move(aes_inputs));
    if (!expect(aes_output->metadata("payload.leakage.byte_indexes") == "[3,5]",
            "PearsonPoiFinder did not forward AES byte-index metadata")) {
        return 1;
    }
    if (!expect(aes_output->metadata("payload.leakage.channels") == "HW(m),HW(y)",
            "PearsonPoiFinder did not forward AES channel metadata")) {
        return 1;
    }
    if (!expect(!aes_output->has_metadata("payload.poi.byte_count"),
            "PearsonPoiFinder still stamped redundant byte-count metadata")) {
        return 1;
    }
    if (!expect(!aes_output->has_metadata("payload.poi.channel_count"),
            "PearsonPoiFinder still stamped redundant channel-count metadata")) {
        return 1;
    }
    if (!expect(!aes_output->has_metadata("payload.poi.target_count"),
            "PearsonPoiFinder still stamped redundant target-count metadata")) {
        return 1;
    }
    if (!expect(aes_output->metadata("payload.poi.features_count") == "3",
            "PearsonPoiFinder did not stamp searched feature count metadata")) {
        return 1;
    }
    if (!expect(!aes_output->has_metadata("payload.poi.target.0.label"),
            "PearsonPoiFinder still stamped old target label metadata")) {
        return 1;
    }
    if (!expect(!aes_output->has_metadata("payload.poi.target.0.channel"),
            "PearsonPoiFinder still stamped old target channel metadata")) {
        return 1;
    }
    const auto aes_payload = aes_output->payload_as<crypto_plugin::CorrelationPoiPayload>();
    if (!expect(aes_payload != nullptr, "PearsonPoiFinder AES payload type was wrong")) {
        return 1;
    }
    if (!expect(aes_payload->result_count() == 2,
            "PearsonPoiFinder AES byte-group count was wrong")) {
        return 1;
    }
    if (!expect(aes_payload->result(0).target_byte_index == 3
                && aes_payload->result(1).target_byte_index == 5,
            "PearsonPoiFinder AES byte indexes were wrong")) {
        return 1;
    }
    if (!expect(aes_payload->result(0).result.dim() == 3
                && aes_payload->result(0).result.size(0) == 2
                && aes_payload->result(0).result.size(1) == 1
                && aes_payload->result(0).result.size(2) == 2,
            "PearsonPoiFinder AES result tensor shape was wrong")) {
        return 1;
    }
    leakflow::SummarySection aes_payload_summary("Payload");
    aes_payload->describe(aes_payload_summary, 1);
    bool saw_byte_3_summary = false;
    bool saw_byte_5_summary = false;
    for (const auto& field : aes_payload_summary.fields) {
        if (field.label == "result" && field.value.text == "(byte_index: 3, shape: [2, 1, 2])") {
            saw_byte_3_summary = true;
        }
        if (field.label == "result" && field.value.text == "(byte_index: 5, shape: [2, 1, 2])") {
            saw_byte_5_summary = true;
        }
    }
    if (!expect(saw_byte_3_summary && saw_byte_5_summary,
            "CorrelationPoiPayload summary did not include per-byte result shapes")) {
        return 1;
    }

    crypto_plugin::CorrelationPoiToPlotAnnotations annotation_converter;
    annotation_converter.set_property("precision", std::int64_t{2});
    const auto annotation_output = annotation_converter.process(aes_output);
    if (!expect(annotation_output.has_value(),
            "CorrelationPoiToPlotAnnotations did not produce an output buffer")) {
        return 1;
    }
    if (!expect(annotation_output->caps().type() == leakflow::base::plot_annotation_caps_type,
            "CorrelationPoiToPlotAnnotations emitted wrong caps type")) {
        return 1;
    }
    if (!expect(annotation_output->metadata("payload.conversion.id")
                == crypto_plugin::correlation_poi_to_plot_annotations_id,
            "CorrelationPoiToPlotAnnotations did not stamp conversion id")) {
        return 1;
    }
    if (!expect(annotation_output->metadata("payload.annotation.kind") == "poi",
            "CorrelationPoiToPlotAnnotations did not stamp annotation kind")) {
        return 1;
    }
    const auto annotations = annotation_output->payload_as<leakflow::base::PlotAnnotationPayload>();
    if (!expect(annotations != nullptr, "CorrelationPoiToPlotAnnotations output payload type was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation_count() == 4,
            "CorrelationPoiToPlotAnnotations annotation count was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).label == "byte_3.HW(m)",
            "CorrelationPoiToPlotAnnotations did not use target label metadata")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).kind == "poi",
            "CorrelationPoiToPlotAnnotations annotation kind was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).sample_index == 0,
            "CorrelationPoiToPlotAnnotations annotation sample_index was wrong")) {
        return 1;
    }
    if (!expect(!annotations->annotation(0).value,
            "CorrelationPoiToPlotAnnotations should not emit exact y values for correlation scores")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).norm_value
                && *annotations->annotation(0).norm_value == 1.0,
            "CorrelationPoiToPlotAnnotations annotation norm_value was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(1).norm_value
                && *annotations->annotation(1).norm_value == -1.0,
            "CorrelationPoiToPlotAnnotations negative correlation norm_value was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).fields.size() == 4,
            "CorrelationPoiToPlotAnnotations annotation field count was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).fields[0].first == "label"
                && annotations->annotation(0).fields[0].second == "HW(m)[3]",
            "CorrelationPoiToPlotAnnotations label field was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).fields[1].first == "key byte"
                && annotations->annotation(0).fields[1].second == "3",
            "CorrelationPoiToPlotAnnotations key-byte field was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).fields[2].first == "target"
                && annotations->annotation(0).fields[2].second == "HW(m)",
            "CorrelationPoiToPlotAnnotations target field was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).fields[3].first == "correlation"
                && annotations->annotation(0).fields[3].second == "1.00",
            "CorrelationPoiToPlotAnnotations score field was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).text.find("byte_3.HW(m):") != std::string::npos,
            "CorrelationPoiToPlotAnnotations annotation text was wrong")) {
        return 1;
    }
    if (!expect(annotations->annotation(0).target_index
                && *annotations->annotation(0).target_index == 0,
            "CorrelationPoiToPlotAnnotations annotation target index was wrong")) {
        return 1;
    }

    leakflow::Pipeline pipeline;
    auto feature_source = pipeline.add(std::make_shared<TensorSource>("features_src", features));
    auto target_source = pipeline.add(std::make_shared<TensorSource>("targets_src", targets));
    auto pipeline_finder_element = std::make_shared<crypto_plugin::PearsonPoiFinder>("poi");
    pipeline_finder_element->set_property("top_k", leakflow::IntList{1});
    auto pipeline_finder = pipeline.add(std::move(pipeline_finder_element));
    pipeline.link(feature_source, "src", pipeline_finder, "features");
    pipeline.link(target_source, "src", pipeline_finder, "targets");
    const auto pipeline_output = pipeline.run();
    if (!expect(pipeline_output.has_value(), "linked multi-sink PearsonPoiFinder pipeline produced no output")) {
        return 1;
    }
    if (!expect(pipeline_output->payload_as<crypto_plugin::CorrelationPoiPayload>() != nullptr,
            "linked multi-sink PearsonPoiFinder pipeline produced wrong payload")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([&features] {
            crypto_plugin::PearsonPoiFinder bad_finder;
            leakflow::ElementInputs bad_inputs;
            bad_inputs.emplace("features", torch_buffer(features));
            (void)bad_finder.process_inputs(std::move(bad_inputs));
        }),
            "PearsonPoiFinder accepted missing targets input")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&features, &targets] {
            crypto_plugin::PearsonPoiFinder bad_finder;
            bad_finder.set_property("top_k", leakflow::IntList{100});
            leakflow::ElementInputs bad_inputs;
            bad_inputs.emplace("features", torch_buffer(features));
            bad_inputs.emplace("targets", torch_buffer(targets));
            (void)bad_finder.process_inputs(std::move(bad_inputs));
        }),
            "PearsonPoiFinder accepted top_k larger than feature count")) {
        return 1;
    }

    const auto descriptors = crypto_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "crypto plugin descriptor count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 3, "crypto plugin element count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[1].type_name == "PearsonPoiFinder",
            "PearsonPoiFinder descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[1].output_pads[0].caps().type() == crypto_plugin::correlation_poi_caps_type,
            "PearsonPoiFinder descriptor output caps were wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[2].type_name == "CorrelationPoiToPlotAnnotations",
            "CorrelationPoiToPlotAnnotations descriptor type name was wrong")) {
        return 1;
    }

    return 0;
}
