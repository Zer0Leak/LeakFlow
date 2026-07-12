#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/plugins/extras/descriptor_catalog.hpp"
#include "leakflow/plugins/extras/extras_elements.hpp"

#include <cnpy++.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
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

template <typename Exception, typename Function>
bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }

    return false;
}

std::filesystem::path temp_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

} // namespace

int main()
{
    namespace extras_plugin = leakflow::plugins::extras;

    const auto traces_path = temp_path("leakflow_numpy_to_torch_traces.npy");
    const auto key_path = temp_path("leakflow_numpy_to_torch_key.npy");
    const std::vector<float> traces{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<std::uint8_t> key{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    cnpypp::npy_save(traces_path.string(), traces.begin(), {2, 3});
    cnpypp::npy_save(key_path.string(), key.begin(), {16});

    extras_plugin::NumpySrc source;
    source.set_property("path", traces_path.string());

    auto numpy_buffer = source.process(std::nullopt);
    if (!expect(numpy_buffer.has_value(), "NumpySrc did not produce a buffer")) {
        return 1;
    }
    numpy_buffer->set_metadata("payload.layout", "trace/sample");

    extras_plugin::NumpyToTorch converter;
    auto output = converter.process(std::move(numpy_buffer));

    if (!expect(output.has_value(), "NumpyToTorch did not produce a buffer")) {
        return 1;
    }
    if (!expect(output->caps().type() == "leakflow/torch-tensor", "NumpyToTorch emitted wrong buffer caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_dtype) == "float32",
                "NumpyToTorch emitted wrong buffer dtype caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_device) == "cpu",
                "NumpyToTorch emitted wrong buffer device caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_rank) == "2",
                "NumpyToTorch emitted wrong buffer rank caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_shape) == "[2,3]",
                "NumpyToTorch emitted wrong buffer shape caps")) {
        return 1;
    }
    if (!expect(output->metadata("origin.file.path") == traces_path.string(), "NumpyToTorch did not preserve file.path")) {
        return 1;
    }
    if (!expect(output->metadata("payload.conversion.id") == extras_plugin::numpy_to_torch_conversion_id,
                "NumpyToTorch did not stamp conversion.id")) {
        return 1;
    }
    if (!expect(output->metadata("payload.conversion.element") == "numpytotorch0",
                "NumpyToTorch did not stamp conversion.element")) {
        return 1;
    }
    if (!expect(output->metadata("payload.layout") == "trace/sample",
                "NumpyToTorch did not preserve the semantic input layout")) {
        return 1;
    }
    if (!expect(converter.input_pads().size() == 1, "NumpyToTorch input pad count changed")) {
        return 1;
    }
    if (!expect(converter.output_pads().size() == 1, "NumpyToTorch output pad count changed")) {
        return 1;
    }
    if (!expect(converter.input_pads()[0].caps().type() == "leakflow/numpy-array",
                "NumpyToTorch input pad caps were wrong")) {
        return 1;
    }
    if (!expect(converter.output_pads()[0].caps().type() == "leakflow/torch-tensor",
                "NumpyToTorch output pad caps were wrong")) {
        return 1;
    }

    const auto payload = output->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(payload != nullptr, "NumpyToTorch payload type was wrong")) {
        return 1;
    }
    if (!expect(payload->dtype() == torch::kFloat32, "NumpyToTorch did not preserve float32 dtype")) {
        return 1;
    }
    if (!expect(payload->shape()[0] == 2 && payload->shape()[1] == 3, "NumpyToTorch generated traces shape mismatch")) {
        return 1;
    }

    extras_plugin::NumpySrc key_source;
    key_source.set_property("path", key_path.string());
    auto key_buffer = key_source.process(std::nullopt);
    extras_plugin::NumpyToTorch float_converter;
    float_converter.set_property("dtype", std::string("float32"));
    auto float_output = float_converter.process(std::move(key_buffer));
    const auto float_payload = float_output->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(float_payload != nullptr, "NumpyToTorch dtype override payload type was wrong")) {
        return 1;
    }
    if (!expect(float_payload->dtype() == torch::kFloat32, "NumpyToTorch dtype override was not applied")) {
        return 1;
    }
    if (!expect(float_output->caps().param(leakflow::base::caps_param_dtype) == "float32",
                "NumpyToTorch dtype override was not reflected in buffer caps")) {
        return 1;
    }
    if (!expect(float_output->caps().param(leakflow::base::caps_param_shape) == "[16]",
                "NumpyToTorch key shape was not reflected in buffer caps")) {
        return 1;
    }
    if (!expect(float_payload->shape()[0] == 16, "NumpyToTorch key fixture shape mismatch")) {
        return 1;
    }
    if (!expect(float_output->metadata("payload.layout") == "axis_0",
                "NumpyToTorch did not preserve the generic key layout")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([&converter] { (void)converter.process(std::nullopt); }),
                "NumpyToTorch accepted missing input")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([&converter] {
                    leakflow::Buffer wrong_caps(leakflow::Caps("leakflow/buffer"));
                    (void)converter.process(std::move(wrong_caps));
                }),
                "NumpyToTorch accepted wrong input caps")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([&converter] {
                    leakflow::Buffer missing_payload(leakflow::Caps("leakflow/numpy-array"));
                    (void)converter.process(std::move(missing_payload));
                }),
                "NumpyToTorch accepted a missing NumpyPayload")) {
        return 1;
    }

    const auto descriptors = extras_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "extras plugin descriptor count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 6, "extras element descriptor count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[1].type_name == "NumpyToTorch",
                "NumpyToTorch descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[1].input_pads[0].caps().type() == "leakflow/numpy-array",
                "NumpyToTorch descriptor input pad caps were wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[1].output_pads[0].caps().type() == "leakflow/torch-tensor",
                "NumpyToTorch descriptor output pad caps were wrong")) {
        return 1;
    }

    std::filesystem::remove(traces_path);
    std::filesystem::remove(key_path);
    return 0;
}
