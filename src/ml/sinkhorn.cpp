#include "leakflow/ml/sinkhorn.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <torch/torch.h>

namespace leakflow::ml {
namespace {

struct PreparedCost {
    torch::Tensor cost; // [U, T, K] float64
    bool batched;
};

[[nodiscard]] PreparedCost prepare_cost(const torch::Tensor& cost)
{
    auto working = cost.to(torch::kFloat64).contiguous();
    if (working.dim() == 2) {
        return {working.unsqueeze(0), false};
    }
    if (working.dim() == 3) {
        return {working, true};
    }
    throw std::invalid_argument("sinkhorn cost must be [T, K] or [U, T, K]");
}

// Broadcast a marginal to [U, length]: accepts [length], [U, length], or (for the unbatched
// caller) a bare vector that is unsqueezed to a single unit.
[[nodiscard]] torch::Tensor prepare_marginal(
    const torch::Tensor& marginal,
    std::int64_t units,
    std::int64_t length,
    const char* what)
{
    auto working = marginal.to(torch::kFloat64).contiguous();
    if (working.dim() == 1) {
        if (working.size(0) != length) {
            throw std::invalid_argument(std::string("sinkhorn ") + what + " has the wrong length");
        }
        return working.unsqueeze(0).expand({units, length}).contiguous();
    }
    if (working.dim() == 2) {
        if (working.size(0) != units || working.size(1) != length) {
            throw std::invalid_argument(std::string("sinkhorn ") + what + " has the wrong shape");
        }
        return working;
    }
    throw std::invalid_argument(std::string("sinkhorn ") + what + " must be 1-D or 2-D");
}

} // namespace

SinkhornResult sinkhorn(
    const torch::Tensor& cost,
    const torch::Tensor& row_marginal,
    const torch::Tensor& col_marginal,
    const SinkhornOptions& options)
{
    if (options.epsilon <= 0.0) {
        throw std::invalid_argument("sinkhorn epsilon must be > 0");
    }
    if (options.max_iter < 1) {
        throw std::invalid_argument("sinkhorn max_iter must be >= 1");
    }

    const auto prepared = prepare_cost(cost);
    const auto& c = prepared.cost; // [U, T, K]
    const auto u = c.size(0);
    const auto t = c.size(1);
    const auto k = c.size(2);

    const auto a = prepare_marginal(row_marginal, u, t, "row_marginal"); // [U, T]
    const auto b = prepare_marginal(col_marginal, u, k, "col_marginal"); // [U, K]

    if ((a <= 0).any().item<bool>() || (b <= 0).any().item<bool>()) {
        throw std::invalid_argument("sinkhorn marginals must be strictly positive");
    }
    if (!torch::allclose(a.sum(-1), b.sum(-1), 1.0e-6, 1.0e-9)) {
        throw std::invalid_argument("sinkhorn requires per-unit sum(row_marginal) == sum(col_marginal)");
    }

    const double eps = options.epsilon;
    const auto log_a = torch::log(a); // [U, T]
    const auto log_b = torch::log(b); // [U, K]

    auto f = torch::zeros({u, t}, c.options()); // [U, T]
    auto g = torch::zeros({u, k}, c.options()); // [U, K]

    std::int64_t iterations = 0;
    for (std::int64_t iteration = 0; iteration < options.max_iter; ++iteration) {
        ++iterations;
        // f-update: logsumexp over K of (g - C)/eps, giving row-normalising potentials.
        f = eps * (log_a - torch::logsumexp((g.unsqueeze(1) - c) / eps, /*dim=*/2));
        // g-update: logsumexp over T of (f - C)/eps, giving column-normalising potentials.
        g = eps * (log_b - torch::logsumexp((f.unsqueeze(2) - c) / eps, /*dim=*/1));

        // Convergence: after the g-update the columns match b by construction, so the
        // residual is the row-marginal drift.
        const auto log_row = torch::logsumexp((f.unsqueeze(2) + g.unsqueeze(1) - c) / eps, /*dim=*/2);
        const auto row_error = (log_row.exp() - a).abs().sum(-1); // [U]
        if (row_error.max().item<double>() < options.tol) {
            break;
        }
    }

    const auto log_plan = (f.unsqueeze(2) + g.unsqueeze(1) - c) / eps; // [U, T, K]
    const auto plan = log_plan.exp();
    const auto row_error = (plan.sum(2) - a).abs().sum(-1); // [U]
    const auto col_error = (plan.sum(1) - b).abs().sum(-1); // [U]
    const auto marginal_error = row_error + col_error;      // [U]

    SinkhornResult result;
    result.batched = prepared.batched;
    result.iterations = iterations;
    if (prepared.batched) {
        result.plan = plan;
        result.log_plan = log_plan;
        result.marginal_error = marginal_error;
    } else {
        result.plan = plan.squeeze(0);
        result.log_plan = log_plan.squeeze(0);
        result.marginal_error = marginal_error.squeeze(0);
    }
    return result;
}

} // namespace leakflow::ml
