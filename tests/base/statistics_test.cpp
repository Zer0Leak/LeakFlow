#include "leakflow/base/statistics.hpp"

#include <iostream>
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

bool expect_close(const torch::Tensor& actual, const torch::Tensor& expected, const char* message)
{
    if (!torch::allclose(actual, expected, 1.0e-5, 1.0e-5)) {
        std::cerr << message << '\n';
        std::cerr << "actual: " << actual << '\n';
        std::cerr << "expected: " << expected << '\n';
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
    const auto columns = torch::tensor(
        {
            {1.0, 1.0, 2.0},
            {2.0, 1.0, 1.0},
            {3.0, 1.0, 0.0},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    const auto targets = torch::tensor(
        {
            {1.0, 3.0},
            {2.0, 2.0},
            {3.0, 1.0},
        },
        torch::TensorOptions().dtype(torch::kFloat32));

    const auto correlation = leakflow::base::pearson_correlation(columns, targets);
    const auto expected = torch::tensor(
        {
            {1.0, 0.0, -1.0},
            {-1.0, 0.0, 1.0},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    if (!expect(correlation.dim() == 2 && correlation.size(0) == 2 && correlation.size(1) == 3,
            "Pearson output shape was wrong")) {
        return 1;
    }
    if (!expect(correlation.scalar_type() == torch::kFloat32, "Pearson default dtype was not input dtype")) {
        return 1;
    }
    if (!expect_close(correlation, expected, "Pearson known-value result was wrong")) {
        return 1;
    }
    if (!expect(!torch::isnan(correlation).any().item<bool>(), "Pearson produced NaN for zero-variance column")) {
        return 1;
    }

    leakflow::base::PearsonCorrelationOptions float64_options;
    float64_options.compute_dtype = leakflow::base::PearsonComputeDtype::Float64;
    const auto float64_correlation = leakflow::base::pearson_correlation(columns, targets, float64_options);
    if (!expect(float64_correlation.scalar_type() == torch::kFloat64, "Pearson float64 option was ignored")) {
        return 1;
    }
    if (!expect_close(float64_correlation.to(torch::kFloat32), expected, "Pearson float64 result was wrong")) {
        return 1;
    }

    const auto batched_columns = columns.reshape({1, 3, 3}).repeat({2, 1, 1});
    const auto batched_targets = targets.reshape({1, 3, 2}).repeat({2, 1, 1});
    const auto batched_correlation = leakflow::base::pearson_correlation(batched_columns, batched_targets);
    if (!expect(batched_correlation.dim() == 3 && batched_correlation.size(0) == 2
                && batched_correlation.size(1) == 2 && batched_correlation.size(2) == 3,
            "Pearson batched output shape was wrong")) {
        return 1;
    }
    if (!expect_close(batched_correlation[0], expected, "Pearson batched first result was wrong")) {
        return 1;
    }
    if (!expect_close(batched_correlation[1], expected, "Pearson batched second result was wrong")) {
        return 1;
    }

    const auto broadcast_targets = targets.reshape({1, 3, 2}).repeat({4, 1, 1});
    const auto broadcast_correlation = leakflow::base::pearson_correlation(columns, broadcast_targets);
    if (!expect(broadcast_correlation.dim() == 3 && broadcast_correlation.size(0) == 4
                && broadcast_correlation.size(1) == 2 && broadcast_correlation.size(2) == 3,
            "Pearson broadcast output shape was wrong")) {
        return 1;
    }
    if (!expect_close(broadcast_correlation[3], expected, "Pearson broadcast result was wrong")) {
        return 1;
    }

    const auto integer_columns = torch::tensor({{1, 2}, {2, 4}, {3, 6}}, torch::TensorOptions().dtype(torch::kInt32));
    const auto integer_targets = torch::tensor({{1}, {2}, {3}}, torch::TensorOptions().dtype(torch::kInt32));
    if (!expect(throws_invalid_argument([&] {
            (void)leakflow::base::pearson_correlation(integer_columns, integer_targets);
        }),
            "Pearson accepted integer columns with input compute dtype")) {
        return 1;
    }

    leakflow::base::PearsonCorrelationOptions float32_options;
    float32_options.compute_dtype = leakflow::base::PearsonComputeDtype::Float32;
    const auto integer_correlation =
        leakflow::base::pearson_correlation(integer_columns, integer_targets, float32_options);
    if (!expect(integer_correlation.scalar_type() == torch::kFloat32,
            "Pearson float32 option did not accept integer inputs")) {
        return 1;
    }
    if (!expect_close(integer_correlation, torch::ones({1, 2}), "Pearson integer conversion result was wrong")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([&columns, &targets] {
            leakflow::base::PearsonCorrelationOptions bad_options;
            bad_options.epsilon = 0.0;
            (void)leakflow::base::pearson_correlation(columns, targets, bad_options);
        }),
            "Pearson accepted non-positive epsilon")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([&columns] {
            (void)leakflow::base::pearson_correlation(columns, torch::ones({2, 2}));
        }),
            "Pearson accepted mismatched row counts")) {
        return 1;
    }

    return 0;
}
