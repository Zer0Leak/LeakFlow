#include "leakflow/core/buffer.hpp"
#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"

#include <iostream>
#include <memory>
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

} // namespace

int main()
{
    auto tensor = torch::arange(6, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3});
    leakflow::base::TorchTensorPayload payload(tensor);

    if (!expect(payload.type_name() == "leakflow/torch-tensor", "tensor payload type name changed")) {
        return 1;
    }
    if (!expect(payload.dtype() == torch::kFloat32, "tensor payload dtype mismatch")) {
        return 1;
    }
    if (!expect(payload.dtype_name() == "float32", "tensor payload dtype name mismatch")) {
        return 1;
    }
    if (!expect(payload.device_name() == "cpu", "tensor payload device name mismatch")) {
        return 1;
    }
    if (!expect(payload.rank() == 2, "tensor payload rank mismatch")) {
        return 1;
    }
    if (!expect(payload.element_count() == 6, "tensor payload element count mismatch")) {
        return 1;
    }
    if (!expect(payload.shape()[0] == 2 && payload.shape()[1] == 3, "tensor payload shape mismatch")) {
        return 1;
    }
    if (!expect(payload.is_cpu(), "CPU tensor was not reported as CPU")) {
        return 1;
    }
    if (!expect(!payload.is_cuda(), "CPU tensor was reported as CUDA")) {
        return 1;
    }
    if (!expect(payload.is_contiguous(), "fresh reshaped tensor should be contiguous")) {
        return 1;
    }
    if (!expect(leakflow::base::torch_dtype_name(torch::kUInt16) == "uint16",
            "uint16 Torch dtype name mismatch")) {
        return 1;
    }
    if (!expect(leakflow::base::torch_dtype_name(torch::kFloat16) == "float16",
            "float16 Torch dtype name mismatch")) {
        return 1;
    }
    if (!expect(leakflow::base::torch_dtype_name(torch::kBFloat16) == "bfloat16",
            "bfloat16 Torch dtype name mismatch")) {
        return 1;
    }
    if (!expect(leakflow::base::torch_dtype_name(torch::kComplexFloat) == "complex64",
            "complex64 Torch dtype name mismatch")) {
        return 1;
    }

    const auto payload_caps = payload.caps();
    if (!expect(payload_caps.type() == leakflow::base::torch_tensor_caps_type, "tensor payload caps type mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_dtype) == "float32",
            "tensor payload caps dtype mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_device) == "cpu",
            "tensor payload caps device mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_rank) == "2",
            "tensor payload caps rank mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_shape) == "[2,3]",
            "tensor payload caps shape mismatch")) {
        return 1;
    }

    leakflow::SummarySection summary_section("Payload");
    payload.describe(summary_section, 2);
    if (!expect(summary_section.fields.size() >= 8, "tensor summary did not include detailed fields")) {
        return 1;
    }
    if (!expect(summary_section.fields[1].label == "dtype", "tensor summary did not include dtype")) {
        return 1;
    }
    if (!expect(summary_section.fields[4].label == "shape", "tensor summary did not include shape")) {
        return 1;
    }

    auto transposed = tensor.transpose(0, 1);
    leakflow::base::TorchTensorPayload non_contiguous_payload(transposed);
    if (!expect(!non_contiguous_payload.is_contiguous(), "non-contiguous strided tensor was not preserved")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            leakflow::base::TorchTensorPayload invalid{torch::Tensor()};
        }),
            "undefined tensors should be rejected")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            const auto indices = torch::tensor({{0, 1}}, torch::TensorOptions().dtype(torch::kInt64));
            const auto values = torch::tensor({1.0F, 2.0F});
            leakflow::base::TorchTensorPayload invalid(torch::sparse_coo_tensor(indices, values, {2}));
        }),
            "non-strided tensors should be rejected")) {
        return 1;
    }

    leakflow::Buffer buffer(leakflow::Caps("leakflow/torch-tensor"));
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(tensor));

    const auto roundtrip = buffer.payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(roundtrip != nullptr, "buffer did not preserve TorchTensorPayload type")) {
        return 1;
    }
    if (!expect(roundtrip->element_count() == 6, "buffer tensor payload element count mismatch")) {
        return 1;
    }

    return 0;
}
