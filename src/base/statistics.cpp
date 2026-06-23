#include "leakflow/base/statistics.hpp"

#include <c10/core/ScalarType.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace leakflow::base {
namespace {

void require_strided_tensor(const torch::Tensor& tensor, std::string_view name)
{
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " tensor must be defined");
    }
    if (tensor.layout() != torch::kStrided) {
        throw std::invalid_argument(std::string(name) + " tensor must use strided layout");
    }
    if (tensor.dim() < 2) {
        throw std::invalid_argument(std::string(name) + " tensor must have at least two dimensions");
    }
}

[[nodiscard]] torch::ScalarType compute_dtype_for(
    torch::ScalarType input_dtype,
    PearsonComputeDtype compute_dtype)
{
    switch (compute_dtype) {
    case PearsonComputeDtype::Input:
        if (!c10::isFloatingType(input_dtype)) {
            throw std::invalid_argument("input Pearson compute dtype requires floating-point columns");
        }
        return input_dtype;
    case PearsonComputeDtype::Float32:
        return torch::kFloat32;
    case PearsonComputeDtype::Float64:
        return torch::kFloat64;
    }

    throw std::invalid_argument("unsupported Pearson compute dtype");
}

[[nodiscard]] bool leading_dimensions_are_broadcastable(const torch::Tensor& columns, const torch::Tensor& targets)
{
    auto column_dim = columns.dim() - 3;
    auto target_dim = targets.dim() - 3;

    while (column_dim >= 0 || target_dim >= 0) {
        const auto column_size = column_dim >= 0 ? columns.size(column_dim) : 1;
        const auto target_size = target_dim >= 0 ? targets.size(target_dim) : 1;
        if (column_size != target_size && column_size != 1 && target_size != 1) {
            return false;
        }
        --column_dim;
        --target_dim;
    }

    return true;
}

void require_matching_shapes(const torch::Tensor& columns, const torch::Tensor& targets)
{
    const auto column_rank = columns.dim();
    const auto target_rank = targets.dim();

    if (columns.device() != targets.device()) {
        throw std::invalid_argument("Pearson inputs must be on the same device");
    }
    if (columns.size(column_rank - 2) != targets.size(target_rank - 2)) {
        throw std::invalid_argument("Pearson inputs must have matching row count");
    }
    if (columns.size(column_rank - 2) < 2) {
        throw std::invalid_argument("Pearson inputs must have at least two rows");
    }
    if (columns.size(column_rank - 1) <= 0) {
        throw std::invalid_argument("Pearson columns input must have at least one column");
    }
    if (targets.size(target_rank - 1) <= 0) {
        throw std::invalid_argument("Pearson targets input must have at least one target");
    }

    if (!leading_dimensions_are_broadcastable(columns, targets)) {
        throw std::invalid_argument("Pearson inputs must have broadcastable leading dimensions");
    }
}

} // namespace

torch::Tensor pearson_correlation(
    const torch::Tensor& columns,
    const torch::Tensor& targets,
    const PearsonCorrelationOptions& options)
{
    require_strided_tensor(columns, "columns");
    require_strided_tensor(targets, "targets");
    require_matching_shapes(columns, targets);

    if (options.epsilon <= 0.0) {
        throw std::invalid_argument("Pearson epsilon must be positive");
    }

    const auto dtype = compute_dtype_for(columns.scalar_type(), options.compute_dtype);
    const auto x = columns.to(dtype);
    const auto y = targets.to(dtype);

    const auto x_centered = x - x.mean(-2, true);
    const auto y_centered = y - y.mean(-2, true);
    const auto numerator = torch::matmul(y_centered.transpose(-2, -1), x_centered);
    const auto x_sum_squares = x_centered.square().sum(-2);
    const auto y_sum_squares = y_centered.square().sum(-2);
    const auto denominator = torch::sqrt(y_sum_squares.unsqueeze(-1) * x_sum_squares.unsqueeze(-2));
    const auto safe_denominator = denominator > options.epsilon;
    const auto correlation = numerator / denominator.clamp_min(options.epsilon);

    return torch::where(safe_denominator, correlation, torch::zeros_like(correlation)).contiguous();
}

} // namespace leakflow::base
