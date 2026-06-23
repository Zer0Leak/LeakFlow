#pragma once

#include <torch/torch.h>

namespace leakflow::base {

enum class PearsonComputeDtype {
    Input,
    Float32,
    Float64,
};

struct PearsonCorrelationOptions {
    PearsonComputeDtype compute_dtype = PearsonComputeDtype::Input;
    double epsilon = 1.0e-12;
};

[[nodiscard]] torch::Tensor pearson_correlation(
    const torch::Tensor& columns,
    const torch::Tensor& targets,
    const PearsonCorrelationOptions& options = {});

} // namespace leakflow::base
