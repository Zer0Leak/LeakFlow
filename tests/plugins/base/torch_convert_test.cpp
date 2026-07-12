#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plugins/base/base_elements.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <torch/torch.h>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

template <typename Exception, typename Function> bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }

    return false;
}

} // namespace

int main()
{
    namespace base_plugin = leakflow::plugins::base;

    auto source_payload = std::make_shared<leakflow::base::TorchTensorPayload>(
        torch::arange(0, 6, torch::TensorOptions().dtype(torch::kInt32)).reshape({2, 3}));
    leakflow::Buffer input(source_payload->caps());
    input.set_metadata("capture.dataset.name", "unit");
    input.set_payload(source_payload);
    input.set_metadata("payload.layout", "trace/sample");

    base_plugin::TorchConvert converter;
    converter.set_property("dtype", std::string("float32"));
    converter.set_property("device", std::string("cpu"));

    auto output = converter.process(input);
    if (!expect(output.has_value(), "TorchConvert did not produce a buffer")) {
        return 1;
    }
    if (!expect(output->caps().type() == leakflow::base::torch_tensor_caps_type,
                "TorchConvert emitted wrong caps type")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_dtype) == "float32",
                "TorchConvert emitted wrong dtype caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_device) == "cpu",
                "TorchConvert emitted wrong device caps")) {
        return 1;
    }
    if (!expect(output->metadata("capture.dataset.name") == "unit", "TorchConvert did not preserve metadata")) {
        return 1;
    }
    if (!expect(output->metadata("payload.conversion.id") == base_plugin::torch_convert_conversion_id,
                "TorchConvert did not stamp conversion id")) {
        return 1;
    }
    if (!expect(output->metadata("payload.layout") == "trace/sample",
                "TorchConvert did not preserve the semantic input layout")) {
        return 1;
    }

    const auto converted_payload = output->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(converted_payload != nullptr, "TorchConvert output payload type was wrong")) {
        return 1;
    }
    if (!expect(converted_payload->dtype() == torch::kFloat32, "TorchConvert did not convert dtype")) {
        return 1;
    }
    if (!expect(converted_payload->is_cpu(), "TorchConvert did not keep tensor on CPU")) {
        return 1;
    }
    if (!expect(torch::equal(converted_payload->tensor(),
                             torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3})),
                "TorchConvert changed tensor values")) {
        return 1;
    }

    base_plugin::TorchConvert empty_input_converter;
    if (!expect(throws_exception<std::invalid_argument>(
                    [&empty_input_converter] { (void)empty_input_converter.process(std::nullopt); }),
                "TorchConvert accepted empty input")) {
        return 1;
    }

    return 0;
}
