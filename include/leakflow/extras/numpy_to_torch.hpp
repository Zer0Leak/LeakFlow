#pragma once

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/extras/numpy_payload.hpp"

#include <optional>
#include <torch/torch.h>

namespace leakflow::extras {

struct NumpyToTorchOptions {
    std::optional<torch::ScalarType> target_dtype = std::nullopt;
    torch::Device target_device = torch::Device(torch::kCPU);
};

[[nodiscard]] leakflow::base::TorchTensorPayload convert_numpy_to_torch(
    const NumpyPayload& payload,
    const NumpyToTorchOptions& options = NumpyToTorchOptions{});

} // namespace leakflow::extras
