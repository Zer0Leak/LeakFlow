#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace leakflow::ml {

// Entropic optimal transport via the log-domain Sinkhorn algorithm (Cuturi, "Sinkhorn
// Distances", NeurIPS 2013), batched over a leading unit axis.
//
// Solves, per unit, P* = argmin_{P in U(a,b)} <P, C> - epsilon * H(P), where
// U(a,b) = { P >= 0 : P 1 = a, P^T 1 = b } and H is the entropy. The regularized optimum
// factorizes as P = diag(u) exp(-C/epsilon) diag(v); we iterate the scalings in the log
// domain (stable for small epsilon) using the dual potentials f, g:
//   f_i <- epsilon ( log a_i - logsumexp_k[(g_k - C_ik)/epsilon] )
//   g_k <- epsilon ( log b_k - logsumexp_i[(f_i - C_ik)/epsilon] )
//
// Motivating use in LeakFlow: the size-constrained GMM E-step (MC labeling), where C is the
// per-component negative log-likelihood, a = ones (each trace assigned once), and b the
// known cluster sizes (the binomial multiset). epsilon dials the assignment from soft
// (large) to hard (small). The primitive itself is domain-agnostic.

struct SinkhornOptions {
    double epsilon = 0.05;         // entropic regularization / temperature (> 0)
    std::int64_t max_iter = 200;   // maximum Sinkhorn iterations
    double tol = 1.0e-7;           // stop when the max per-unit marginal violation < tol
};

struct SinkhornResult {
    torch::Tensor plan;           // [U, T, K] (or [T, K]) transport plan P >= 0
    torch::Tensor log_plan;       // [U, T, K] (or [T, K]) log P
    torch::Tensor marginal_error; // [U] (or scalar) final ||P 1 - a||_1 + ||P^T 1 - b||_1
    std::int64_t iterations = 0;  // iterations run until all units met tol (or max_iter)
    bool batched = true;          // false when the caller passed unbatched [T, K]
};

// cost           C: [T, K] or [U, T, K]  (lower = better match)
// row_marginal   a: [T]    or [U, T]      (target row sums of P; per-unit sum must match b)
// col_marginal   b: [K]    or [U, K]      (target column sums of P)
// All computation is float64. Requires per-unit sum(a) == sum(b) and strictly positive
// marginals. Throws std::invalid_argument on shape/mass mismatch.
[[nodiscard]] SinkhornResult sinkhorn(
    const torch::Tensor& cost,
    const torch::Tensor& row_marginal,
    const torch::Tensor& col_marginal,
    const SinkhornOptions& options = {});

} // namespace leakflow::ml
