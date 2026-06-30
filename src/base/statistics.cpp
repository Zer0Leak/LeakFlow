#include "leakflow/base/statistics.hpp"

#include <c10/core/ScalarType.h>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

void require_interactive_shapes(const torch::Tensor& columns, const torch::Tensor& targets)
{
    if (columns.dim() != 2) {
        throw std::invalid_argument("interactive Pearson columns must have shape [N, S]");
    }
    if (targets.dim() < 2) {
        throw std::invalid_argument("interactive Pearson targets must have shape [..., N, T]");
    }
    if (columns.device() != targets.device()) {
        throw std::invalid_argument("interactive Pearson inputs must be on the same device");
    }
    if (columns.size(0) != targets.size(-2)) {
        throw std::invalid_argument("interactive Pearson inputs must have matching row count");
    }
    if (columns.size(0) <= 0) {
        throw std::invalid_argument("interactive Pearson inputs must have at least one row");
    }
    if (columns.size(1) <= 0) {
        throw std::invalid_argument("interactive Pearson columns input must have at least one column");
    }
    if (targets.size(-1) <= 0) {
        throw std::invalid_argument("interactive Pearson targets input must have at least one target");
    }
}

[[nodiscard]] std::vector<std::int64_t> target_result_shape(const torch::Tensor& targets)
{
    auto shape = targets.sizes().vec();
    shape.erase(shape.end() - 2);
    return shape;
}

[[nodiscard]] std::string shape_string(const std::vector<std::int64_t>& shape)
{
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
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

InteractivePearsonCorrelation::InteractivePearsonCorrelation(PearsonCorrelationOptions options)
    : options_(std::move(options))
{
    if (options_.epsilon <= 0.0) {
        throw std::invalid_argument("interactive Pearson epsilon must be positive");
    }
}

torch::Tensor InteractivePearsonCorrelation::update(const torch::Tensor& columns, const torch::Tensor& targets)
{
    require_strided_tensor(columns, "columns");
    require_strided_tensor(targets, "targets");
    require_interactive_shapes(columns, targets);

    const auto batch_count = columns.size(0);
    if (observation_count_ > std::numeric_limits<std::int64_t>::max() - batch_count) {
        throw std::overflow_error("interactive Pearson observation count overflow");
    }

    const auto dtype = compute_dtype_for(columns.scalar_type(), options_.compute_dtype);
    const auto current_target_result_shape = target_result_shape(targets);

    if (!empty()) {
        if (columns.device() != column_mean_.device()) {
            throw std::invalid_argument("interactive Pearson updates must remain on the same device");
        }
        if (dtype != column_mean_.scalar_type()) {
            throw std::invalid_argument("interactive Pearson updates must use the same compute dtype");
        }
        if (columns.size(1) != column_mean_.size(0)) {
            throw std::invalid_argument("interactive Pearson updates must keep the same column count");
        }
        if (current_target_result_shape != target_result_shape_) {
            throw std::invalid_argument("interactive Pearson updates must keep target shape " +
                                        shape_string(target_result_shape_) + ", received " +
                                        shape_string(current_target_result_shape));
        }
    }

    const auto x = columns.to(dtype);
    const auto y = targets.to(dtype);
    const auto batch_column_mean = x.mean(0);
    const auto batch_target_mean = y.mean(-2);
    const auto centered_columns = x - batch_column_mean;
    const auto centered_targets = y - batch_target_mean.unsqueeze(-2);
    const auto batch_column_sum_squares = centered_columns.square().sum(0);
    const auto batch_target_sum_squares = centered_targets.square().sum(-2);
    const auto batch_co_moment = torch::matmul(centered_targets.transpose(-2, -1), centered_columns);

    if (empty()) {
        observation_count_ = batch_count;
        target_result_shape_ = current_target_result_shape;
        column_mean_ = batch_column_mean;
        column_sum_squares_ = batch_column_sum_squares;
        target_mean_ = batch_target_mean;
        target_sum_squares_ = batch_target_sum_squares;
        co_moment_ = batch_co_moment;
        return correlation();
    }

    const auto total_count = observation_count_ + batch_count;
    const auto merge_weight =
        static_cast<double>(observation_count_) * static_cast<double>(batch_count) / static_cast<double>(total_count);
    const auto batch_fraction = static_cast<double>(batch_count) / static_cast<double>(total_count);
    const auto column_mean_delta = batch_column_mean - column_mean_;
    const auto target_mean_delta = batch_target_mean - target_mean_;

    co_moment_ =
        co_moment_ + batch_co_moment + target_mean_delta.unsqueeze(-1) * column_mean_delta.unsqueeze(-2) * merge_weight;
    column_sum_squares_ = column_sum_squares_ + batch_column_sum_squares + column_mean_delta.square() * merge_weight;
    target_sum_squares_ = target_sum_squares_ + batch_target_sum_squares + target_mean_delta.square() * merge_weight;
    column_mean_ = column_mean_ + column_mean_delta * batch_fraction;
    target_mean_ = target_mean_ + target_mean_delta * batch_fraction;
    observation_count_ = total_count;

    return correlation();
}

torch::Tensor InteractivePearsonCorrelation::correlation() const
{
    if (empty()) {
        throw std::logic_error("interactive Pearson correlation has no observations");
    }

    const auto denominator = torch::sqrt(target_sum_squares_.unsqueeze(-1) * column_sum_squares_.unsqueeze(-2));
    const auto safe_denominator = denominator > options_.epsilon;
    const auto value = co_moment_ / denominator.clamp_min(options_.epsilon);
    return torch::where(safe_denominator, value, torch::zeros_like(value)).contiguous();
}

void InteractivePearsonCorrelation::reset()
{
    observation_count_ = 0;
    target_result_shape_.clear();
    column_mean_ = torch::Tensor();
    column_sum_squares_ = torch::Tensor();
    target_mean_ = torch::Tensor();
    target_sum_squares_ = torch::Tensor();
    co_moment_ = torch::Tensor();
}

std::int64_t InteractivePearsonCorrelation::observation_count() const noexcept
{
    return observation_count_;
}

bool InteractivePearsonCorrelation::empty() const noexcept
{
    return observation_count_ == 0;
}

} // namespace leakflow::base
