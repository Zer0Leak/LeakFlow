#include "leakflow/ml/sinkhorn.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
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

[[nodiscard]] double transport_cost(const torch::Tensor& plan, const torch::Tensor& cost)
{
    return (plan * cost).sum().item<double>();
}

} // namespace

int main()
{
    torch::manual_seed(7);

    // --- Marginal recovery on a random cost ---
    {
        constexpr std::int64_t t = 40;
        constexpr std::int64_t k = 6;
        const auto cost = torch::rand({t, k}, torch::kFloat64);
        const auto a = torch::ones({t}, torch::kFloat64);
        auto b = torch::rand({k}, torch::kFloat64) + 0.1;
        b = b * (static_cast<double>(t) / b.sum()); // match total mass

        leakflow::ml::SinkhornOptions options;
        options.epsilon = 0.05;
        options.max_iter = 500;
        const auto result = leakflow::ml::sinkhorn(cost, a, b, options);

        if (!expect(!result.batched, "recovery: expected unbatched result")) {
            return 1;
        }
        if (!expect(result.plan.sizes() == torch::IntArrayRef({t, k}), "recovery: plan shape wrong")) {
            return 1;
        }
        if (!expect((result.plan >= 0).all().item<bool>(), "recovery: plan has negative entries")) {
            return 1;
        }
        if (!expect(torch::allclose(result.plan, result.log_plan.exp(), 1.0e-9, 1.0e-12),
                    "recovery: plan != exp(log_plan)")) {
            return 1;
        }
        if (!expect(torch::allclose(result.plan.sum(1), a, 0.0, 1.0e-4), "recovery: row sums != a")) {
            return 1;
        }
        if (!expect(torch::allclose(result.plan.sum(0), b, 0.0, 1.0e-4), "recovery: col sums != b")) {
            return 1;
        }
    }

    // --- Uniform cost => independent coupling P = a b^T / total (analytic) ---
    {
        constexpr std::int64_t t = 10;
        constexpr std::int64_t k = 4;
        const auto cost = torch::zeros({t, k}, torch::kFloat64);
        const auto a = torch::ones({t}, torch::kFloat64);
        auto b = torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::kFloat64);
        b = b * (static_cast<double>(t) / b.sum());
        const auto result = leakflow::ml::sinkhorn(cost, a, b, {});
        const auto expected = torch::outer(a, b) / a.sum();
        if (!expect(torch::allclose(result.plan, expected, 0.0, 1.0e-6),
                    "uniform: plan != outer(a,b)/total")) {
            return 1;
        }
    }

    // --- epsilon sharpens toward the assignment optimum ---
    {
        constexpr std::int64_t n = 4;
        const auto cost = 1.0 - torch::eye(n, torch::kFloat64); // diagonal 0, off-diagonal 1
        const auto a = torch::ones({n}, torch::kFloat64);
        const auto b = torch::ones({n}, torch::kFloat64);

        leakflow::ml::SinkhornOptions soft;
        soft.epsilon = 1.0;
        leakflow::ml::SinkhornOptions sharp;
        sharp.epsilon = 0.01;
        sharp.max_iter = 2000;
        const auto soft_plan = leakflow::ml::sinkhorn(cost, a, b, soft).plan;
        const auto sharp_plan = leakflow::ml::sinkhorn(cost, a, b, sharp).plan;

        // The optimal doubly-stochastic assignment is the identity (cost 0).
        if (!expect(transport_cost(sharp_plan, cost) < 0.1, "epsilon: sharp plan not near optimum")) {
            return 1;
        }
        if (!expect(transport_cost(sharp_plan, cost) < transport_cost(soft_plan, cost),
                    "epsilon: smaller epsilon did not lower transport cost")) {
            return 1;
        }
        if (!expect(torch::allclose(sharp_plan, torch::eye(n, torch::kFloat64), 0.0, 0.05),
                    "epsilon: sharp plan not ~identity")) {
            return 1;
        }
    }

    // --- Batched over U matches per-unit unbatched runs ---
    {
        constexpr std::int64_t u = 2;
        constexpr std::int64_t t = 15;
        constexpr std::int64_t k = 5;
        const auto cost = torch::rand({u, t, k}, torch::kFloat64);
        const auto a = torch::ones({u, t}, torch::kFloat64);
        auto b = torch::rand({u, k}, torch::kFloat64) + 0.1;
        b = b * (static_cast<double>(t) / b.sum(-1, /*keepdim=*/true));

        leakflow::ml::SinkhornOptions options;
        options.epsilon = 0.1;
        options.max_iter = 500;
        const auto batched = leakflow::ml::sinkhorn(cost, a, b, options);
        if (!expect(batched.batched && batched.plan.sizes() == torch::IntArrayRef({u, t, k}),
                    "batched: shape wrong")) {
            return 1;
        }
        for (std::int64_t unit = 0; unit < u; ++unit) {
            const auto single = leakflow::ml::sinkhorn(cost[unit], a[unit], b[unit], options);
            if (!expect(torch::allclose(batched.plan[unit], single.plan, 1.0e-7, 1.0e-9),
                        "batched: per-unit plan differs from unbatched")) {
                return 1;
            }
        }
    }

    // --- Error paths ---
    if (!expect(throws_invalid_argument([] {
                    (void)leakflow::ml::sinkhorn(
                        torch::rand({4, 3}), torch::ones({4}), torch::ones({3}) * 2.0, {});
                }),
                "expected invalid_argument for mismatched marginal totals")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
                    auto b = torch::ones({3});
                    b[0] = -1.0;
                    (void)leakflow::ml::sinkhorn(torch::rand({3, 3}), torch::ones({3}), b, {});
                }),
                "expected invalid_argument for non-positive marginal")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
                    leakflow::ml::SinkhornOptions bad;
                    bad.epsilon = 0.0;
                    (void)leakflow::ml::sinkhorn(torch::rand({3, 3}), torch::ones({3}), torch::ones({3}), bad);
                }),
                "expected invalid_argument for epsilon <= 0")) {
        return 1;
    }

    std::cout << "sinkhorn tests passed\n";
    return 0;
}
