#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/extras/numpy_payload.hpp"
#include "leakflow/extras/numpy_to_torch.hpp"

#include <cnpy++.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
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
    const auto traces_path = temp_path("leakflow_extras_numpy_to_torch_traces.npy");
    const auto key_path = temp_path("leakflow_extras_numpy_to_torch_key.npy");
    const std::vector<float> traces_data{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<std::uint8_t> key_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    cnpypp::npy_save(traces_path.string(), traces_data.begin(), {2, 3});
    cnpypp::npy_save(key_path.string(), key_data.begin(), {16});

    auto traces = leakflow::extras::load_npy(traces_path);
    auto traces_tensor_payload = leakflow::extras::convert_numpy_to_torch(traces);

    if (!expect(traces_tensor_payload.type_name() == "leakflow/torch-tensor",
                "converted traces payload type name changed")) {
        return 1;
    }
    if (!expect(traces_tensor_payload.dtype() == torch::kFloat32, "float32 NumPy dtype was not preserved")) {
        return 1;
    }
    if (!expect(traces_tensor_payload.is_cpu(), "default conversion should produce a CPU tensor")) {
        return 1;
    }
    if (!expect(traces_tensor_payload.rank() == 2, "converted traces rank mismatch")) {
        return 1;
    }
    if (!expect(traces_tensor_payload.shape()[0] == 2 && traces_tensor_payload.shape()[1] == 3,
                "converted traces shape mismatch")) {
        return 1;
    }
    const auto traces_caps = traces_tensor_payload.caps();
    if (!expect(traces_caps.type() == leakflow::base::torch_tensor_caps_type, "converted traces caps type mismatch")) {
        return 1;
    }
    if (!expect(traces_caps.param(leakflow::base::caps_param_dtype) == "float32",
                "converted traces caps dtype mismatch")) {
        return 1;
    }
    if (!expect(traces_caps.param(leakflow::base::caps_param_device) == "cpu",
                "converted traces caps device mismatch")) {
        return 1;
    }
    if (!expect(traces_caps.param(leakflow::base::caps_param_shape) == "[2,3]",
                "converted traces caps shape mismatch")) {
        return 1;
    }

    const auto expected_trace_value = traces.array().data<float>()[4];
    if (!expect(traces_tensor_payload.tensor().data_ptr<float>()[4] == expected_trace_value,
                "converted traces value mismatch")) {
        return 1;
    }

    auto key = leakflow::extras::load_npy(key_path);
    auto key_tensor_payload = leakflow::extras::convert_numpy_to_torch(key);

    if (!expect(key_tensor_payload.dtype() == torch::kUInt8, "uint8 NumPy dtype was not preserved")) {
        return 1;
    }
    if (!expect(key_tensor_payload.rank() == 1, "converted key rank mismatch")) {
        return 1;
    }
    if (!expect(key_tensor_payload.shape()[0] == 16, "converted key shape mismatch")) {
        return 1;
    }

    const auto original_key_byte = key.array().data<std::uint8_t>()[0];
    if (!expect(key_tensor_payload.tensor().data_ptr<std::uint8_t>()[0] == original_key_byte,
                "converted key value mismatch")) {
        return 1;
    }

    key.array().data<std::uint8_t>()[0] = static_cast<std::uint8_t>(original_key_byte + 1);
    if (!expect(key_tensor_payload.tensor().data_ptr<std::uint8_t>()[0] == original_key_byte,
                "converted tensor should own storage independent of the NumPy "
                "payload")) {
        return 1;
    }

    leakflow::extras::NumpyToTorchOptions float_options;
    float_options.target_dtype = torch::kFloat32;
    auto key_float_payload = leakflow::extras::convert_numpy_to_torch(key, float_options);
    if (!expect(key_float_payload.dtype() == torch::kFloat32, "target dtype override was not applied")) {
        return 1;
    }
    if (!expect(key_float_payload.tensor().data_ptr<float>()[0] ==
                    static_cast<float>(key.array().data<std::uint8_t>()[0]),
                "target dtype conversion value mismatch")) {
        return 1;
    }

    leakflow::Buffer buffer(leakflow::Caps("leakflow/torch-tensor"));
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(key_float_payload)));
    const auto roundtrip = buffer.payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(roundtrip != nullptr, "buffer did not preserve converted TorchTensorPayload type")) {
        return 1;
    }
    if (!expect(roundtrip->dtype() == torch::kFloat32, "buffer converted tensor dtype mismatch")) {
        return 1;
    }

    leakflow::extras::NumpyPayload missing_dtype(cnpypp::npy_load(key_path.string()));
    if (!expect(throws_exception<std::invalid_argument>(
                    [&missing_dtype] { (void)leakflow::extras::convert_numpy_to_torch(missing_dtype); }),
                "conversion should reject NumPy payloads without dtype metadata")) {
        return 1;
    }

    std::filesystem::remove(traces_path);
    std::filesystem::remove(key_path);
    return 0;
}
