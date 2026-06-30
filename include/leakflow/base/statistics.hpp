#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <vector>

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

class InteractivePearsonCorrelation {
public:
    explicit InteractivePearsonCorrelation(PearsonCorrelationOptions options = {});

    [[nodiscard]] torch::Tensor update(const torch::Tensor& columns, const torch::Tensor& targets);
    [[nodiscard]] torch::Tensor correlation() const;

    void reset();

    [[nodiscard]] std::int64_t observation_count() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

private:
    PearsonCorrelationOptions options_;
    std::int64_t observation_count_ = 0;
    std::vector<std::int64_t> target_result_shape_;
    torch::Tensor column_mean_;
    torch::Tensor column_sum_squares_;
    torch::Tensor target_mean_;
    torch::Tensor target_sum_squares_;
    torch::Tensor co_moment_;
};

} // namespace leakflow::base
