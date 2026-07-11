#include "leakflow/ml/gaussian_mixture.hpp"

#include "leakflow/ml/sinkhorn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace leakflow::ml {
namespace {

constexpr double kTwoPi = 2.0 * M_PI;
constexpr double kNkFloor = 1.0e-10; // sklearn's guard so empty components don't divide by zero

// Working layout throughout is [U, T, N]: U attack units (batch), T samples, N features.
struct PreparedInput {
    torch::Tensor x; // [U, T, N] float64
    bool batched;    // false when the caller passed [T, N]
};

[[nodiscard]] PreparedInput prepare_input(const torch::Tensor& x)
{
    auto working = x.to(torch::kFloat64).contiguous();
    if (working.dim() == 2) {
        return {working.unsqueeze(0), false};
    }
    if (working.dim() == 3) {
        return {working, true};
    }
    throw std::invalid_argument("GaussianMixture expects a [T, N] or [U, T, N] tensor");
}

// Drop the leading unit axis when the caller worked with a single unbatched matrix.
[[nodiscard]] torch::Tensor drop_unit_axis(const torch::Tensor& tensor, bool batched)
{
    return batched ? tensor : tensor.squeeze(0);
}

[[nodiscard]] torch::Tensor identity_like(const torch::Tensor& covariance, std::int64_t n)
{
    return torch::eye(n, covariance.options());
}

// Cholesky with jitter: add progressively more to the diagonal of any batch element that
// is not positive-definite. reg_covar has already been folded into the covariance; this is
// the last-resort guard sklearn also relies on for near-singular components.
[[nodiscard]] torch::Tensor robust_cholesky(const torch::Tensor& covariance, double reg_covar)
{
    const auto n = covariance.size(-1);
    auto matrix = covariance.clone();
    const auto eye = identity_like(covariance, n);
    for (int attempt = 0; attempt < 4; ++attempt) {
        auto [factor, info] = torch::linalg_cholesky_ex(matrix, /*upper=*/false, /*check_errors=*/false);
        if (info.eq(0).all().item<bool>()) {
            return factor;
        }
        const double jitter = std::max(reg_covar, 1.0e-12) * std::pow(10.0, attempt + 1);
        matrix = matrix + jitter * eye;
    }
    throw std::runtime_error(
        "GaussianMixture: a component covariance stayed non-positive-definite; raise reg_covar");
}

// log N(x | mean_k, cov_k) for every component, returned as [U, T, K]. Loops over K so the
// largest temporary is [U, T, N] rather than a full [U, T, K, N].
[[nodiscard]] torch::Tensor estimate_log_gaussian_prob(
    const torch::Tensor& x,              // [U, T, N]
    const torch::Tensor& means,          // [U, K, N]
    const torch::Tensor& covariances,    // full: [U,K,N,N]; diag: [U,K,N]
    GmmCovarianceType covariance_type,
    double reg_covar)
{
    const auto n = x.size(2);
    const auto k_count = means.size(1);
    const double constant = static_cast<double>(n) * std::log(kTwoPi);

    std::vector<torch::Tensor> per_component;
    per_component.reserve(static_cast<std::size_t>(k_count));
    for (std::int64_t k = 0; k < k_count; ++k) {
        const auto diff = x - means.select(1, k).unsqueeze(1); // [U, T, N]
        torch::Tensor log_det;                                 // [U]
        torch::Tensor mahalanobis;                             // [U, T]
        if (covariance_type == GmmCovarianceType::Full) {
            const auto factor = robust_cholesky(covariances.select(1, k), reg_covar); // [U, N, N]
            log_det = 2.0 * torch::log(torch::diagonal(factor, 0, -2, -1)).sum(-1);   // [U]
            const auto solved = torch::linalg_solve_triangular(
                factor, diff.transpose(-1, -2), /*upper=*/false, /*left=*/true, /*unitriangular=*/false);
            mahalanobis = solved.pow(2).sum(-2); // [U, T]
        } else {
            const auto variance = covariances.select(1, k);        // [U, N]
            log_det = torch::log(variance).sum(-1).unsqueeze(-1);  // [U, 1]
            mahalanobis = (diff.pow(2) / variance.unsqueeze(1)).sum(-1); // [U, T]
        }
        const auto log_det_row = log_det.dim() == 1 ? log_det.unsqueeze(-1) : log_det; // [U, 1]
        per_component.push_back(-0.5 * (constant + log_det_row + mahalanobis));         // [U, T]
    }
    return torch::stack(per_component, 2); // [U, T, K]
}

struct EStep {
    torch::Tensor log_prob_norm; // [U, T]
    torch::Tensor log_resp;      // [U, T, K]
};

[[nodiscard]] EStep e_step(
    const torch::Tensor& x,
    const torch::Tensor& weights,     // [U, K]
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    GmmCovarianceType covariance_type,
    double reg_covar)
{
    const auto log_prob = estimate_log_gaussian_prob(x, means, covariances, covariance_type, reg_covar);
    const auto weighted = log_prob + torch::log(weights).unsqueeze(1); // [U, T, K]
    const auto log_prob_norm = torch::logsumexp(weighted, -1);         // [U, T]
    const auto log_resp = weighted - log_prob_norm.unsqueeze(-1);      // [U, T, K]
    return {log_prob_norm, log_resp};
}

struct MixtureParameters {
    torch::Tensor weights;     // [U, K]
    torch::Tensor means;       // [U, K, N]
    torch::Tensor covariances; // full: [U,K,N,N]; diag: [U,K,N]
};

// M-step from responsibilities (not log-responsibilities). reg_covar is added to covariance
// diagonals, matching sklearn.
[[nodiscard]] MixtureParameters m_step(
    const torch::Tensor& x,     // [U, T, N]
    const torch::Tensor& resp,  // [U, T, K]
    GmmCovarianceType covariance_type,
    double reg_covar)
{
    const auto u = x.size(0);
    const auto n = x.size(2);
    const auto nk = resp.sum(1) + kNkFloor;             // [U, K]
    const auto weights = nk / nk.sum(-1, /*keepdim=*/true);
    const auto means = torch::einsum("utk,utn->ukn", {resp, x}) / nk.unsqueeze(-1); // [U, K, N]

    const auto k_count = means.size(1);
    std::vector<torch::Tensor> per_component;
    per_component.reserve(static_cast<std::size_t>(k_count));
    for (std::int64_t k = 0; k < k_count; ++k) {
        const auto diff = x - means.select(1, k).unsqueeze(1); // [U, T, N]
        const auto rk = resp.select(2, k);                     // [U, T]
        if (covariance_type == GmmCovarianceType::Full) {
            const auto weighted = diff * rk.unsqueeze(-1);                       // [U, T, N]
            auto covariance = torch::einsum("utn,utm->unm", {weighted, diff})    // [U, N, N]
                / nk.select(1, k).view({u, 1, 1});
            covariance = covariance + reg_covar * identity_like(covariance, n);
            per_component.push_back(covariance);
        } else {
            auto variance = (rk.unsqueeze(-1) * diff.pow(2)).sum(1) / nk.select(1, k).unsqueeze(-1); // [U, N]
            variance = variance + reg_covar;
            per_component.push_back(variance);
        }
    }
    return {weights, means, torch::stack(per_component, 1)};
}

// Squared Euclidean distances from every sample to every centre: [U, T, K].
[[nodiscard]] torch::Tensor squared_distances(const torch::Tensor& x, const torch::Tensor& centres)
{
    const auto distances = torch::cdist(x, centres); // [U, T, K]
    return distances.pow(2);
}

// Gather one row per unit: x is [U, T, N], indexes [U] -> [U, N].
[[nodiscard]] torch::Tensor gather_rows(const torch::Tensor& x, const torch::Tensor& indexes)
{
    const auto u = x.size(0);
    const auto n = x.size(2);
    return x.gather(1, indexes.view({u, 1, 1}).expand({u, 1, n})).squeeze(1);
}

// k-means++ seeding (batched over U), then a few Lloyd iterations. Returns centres [U, K, N].
[[nodiscard]] torch::Tensor kmeans_plus_plus(
    const torch::Tensor& x,
    std::int64_t k_count,
    torch::Generator& generator,
    std::int64_t lloyd_iters)
{
    const auto u = x.size(0);
    const auto t = x.size(1);
    const auto n = x.size(2);

    const auto first = torch::randint(t, {u}, generator, torch::TensorOptions().dtype(torch::kLong));
    std::vector<torch::Tensor> centres_list;
    centres_list.push_back(gather_rows(x, first)); // [U, N]

    for (std::int64_t c = 1; c < k_count; ++c) {
        const auto centres = torch::stack(centres_list, 1);        // [U, c, N]
        const auto nearest = std::get<0>(squared_distances(x, centres).min(2)); // [U, T]
        const auto totals = nearest.sum(-1, /*keepdim=*/true);     // [U, 1]
        // Fall back to uniform sampling for a degenerate (all-coincident) unit.
        auto probabilities = torch::where(totals > 0, nearest / totals, torch::full_like(nearest, 1.0 / t));
        const auto chosen = torch::multinomial(probabilities, 1, /*replacement=*/false, generator).squeeze(1); // [U]
        centres_list.push_back(gather_rows(x, chosen));
    }

    auto centres = torch::stack(centres_list, 1); // [U, K, N]
    for (std::int64_t iter = 0; iter < lloyd_iters; ++iter) {
        const auto labels = squared_distances(x, centres).argmin(2);                 // [U, T]
        const auto onehot = torch::one_hot(labels, k_count).to(x.dtype());           // [U, T, K]
        const auto counts = onehot.sum(1);                                           // [U, K]
        const auto sums = torch::einsum("utk,utn->ukn", {onehot, x});                // [U, K, N]
        const auto updated = sums / counts.clamp_min(1.0).unsqueeze(-1);             // [U, K, N]
        centres = torch::where(counts.unsqueeze(-1) > 0, updated, centres);          // keep empties
    }
    return centres;
}

// Initial responsibilities [U, T, K]; the caller runs one M-step to turn these into params.
[[nodiscard]] torch::Tensor initial_responsibilities(
    const torch::Tensor& x,
    std::int64_t k_count,
    GmmInitMethod method,
    torch::Generator& generator)
{
    const auto u = x.size(0);
    const auto t = x.size(1);
    switch (method) {
    case GmmInitMethod::Random: {
        auto resp = torch::rand({u, t, k_count}, generator, x.options());
        return resp / resp.sum(-1, /*keepdim=*/true);
    }
    case GmmInitMethod::RandomFromData: {
        // Distinct random seed rows per unit via per-unit argsort of noise.
        const auto order = torch::rand({u, t}, generator, x.options()).argsort(/*dim=*/1, /*descending=*/false); // [U, T]
        const auto picks = order.narrow(1, 0, k_count);                            // [U, K]
        std::vector<torch::Tensor> centre_rows;
        centre_rows.reserve(static_cast<std::size_t>(k_count));
        for (std::int64_t k = 0; k < k_count; ++k) {
            centre_rows.push_back(gather_rows(x, picks.select(1, k)));
        }
        const auto centres = torch::stack(centre_rows, 1);              // [U, K, N]
        const auto labels = squared_distances(x, centres).argmin(2);   // [U, T]
        return torch::one_hot(labels, k_count).to(x.dtype());
    }
    case GmmInitMethod::KMeansPlusPlus:
    default: {
        const auto centres = kmeans_plus_plus(x, k_count, generator, /*lloyd_iters=*/10);
        const auto labels = squared_distances(x, centres).argmin(2);
        return torch::one_hot(labels, k_count).to(x.dtype());
    }
    }
}

// where(mask, chosen, current) with the [U] mask broadcast across chosen's trailing axes.
[[nodiscard]] torch::Tensor pick_where(
    const torch::Tensor& unit_mask,
    const torch::Tensor& chosen,
    const torch::Tensor& current)
{
    auto shape = std::vector<std::int64_t>(static_cast<std::size_t>(chosen.dim()), 1);
    shape[0] = unit_mask.size(0);
    return torch::where(unit_mask.view(shape), chosen, current);
}

// Bind the target size multiset to components by mass rank (the 1-D optimal-transport /
// monotone matching): the heaviest component gets the largest size. Returns per-unit counts
// [U, K] summing to T, regardless of whether target_sizes were counts or proportions.
[[nodiscard]] torch::Tensor bind_sizes_by_mass(
    const torch::Tensor& weights,      // [U, K]
    const torch::Tensor& target_sizes, // [K] or [U, K]
    std::int64_t t)
{
    const auto u = weights.size(0);
    const auto k = weights.size(1);
    auto sizes = target_sizes.to(torch::kFloat64);
    if (sizes.dim() == 1) {
        sizes = sizes.unsqueeze(0).expand({u, k});
    }
    const auto sorted_sizes = std::get<0>(sizes.sort(/*dim=*/1, /*descending=*/true)); // [U, K]
    const auto mass_order = weights.argsort(/*dim=*/1, /*descending=*/true);            // [U, K]
    auto bound = torch::zeros({u, k}, torch::kFloat64);
    bound.scatter_(1, mass_order, sorted_sizes); // component mass_order[u,j] gets sorted_sizes[u,j]
    return bound * (static_cast<double>(t) / bound.sum(1, /*keepdim=*/true));
}

struct ConstrainedOutcome {
    torch::Tensor responsibilities; // [U, T, K] final Sinkhorn plan
    torch::Tensor log_prob_norm;    // [U, T] GMM density score under the pinned weights
    torch::Tensor converged;        // [U] bool
    torch::Tensor n_iter;           // [U] int64
};

// Size-constrained EM (Genevay et al. 2019): warm-started params in/out. Each outer step
// replaces the softmax E-step with a Sinkhorn projection (Cuturi 2013) whose column marginal
// is the bound sizes, then runs the ordinary M-step. Weights end pinned to the sizes.
[[nodiscard]] ConstrainedOutcome run_constrained_em(
    const torch::Tensor& x,
    torch::Tensor& weights,     // in/out [U, K]
    torch::Tensor& means,       // in/out [U, K, N]
    torch::Tensor& covariances, // in/out
    const torch::Tensor& target_sizes,
    const GaussianMixtureOptions& options,
    GmmCovarianceType cov_type,
    double reg,
    const GmmProgressCallback& on_progress,
    double frac_base,
    double frac_span)
{
    const auto u = x.size(0);
    const auto t = x.size(1);
    const auto b = bind_sizes_by_mass(weights, target_sizes, t); // [U, K]
    const auto a = torch::ones({u, t}, x.options());             // [U, T]

    SinkhornOptions sinkhorn_options;
    sinkhorn_options.epsilon = options.sinkhorn_epsilon;
    sinkhorn_options.max_iter = options.sinkhorn_max_iter;

    auto converged = torch::zeros({u}, torch::TensorOptions().dtype(torch::kBool));
    auto n_iter = torch::full({u}, options.constrained_max_iter, torch::TensorOptions().dtype(torch::kLong));

    for (std::int64_t outer = 1; outer <= options.constrained_max_iter; ++outer) {
        const auto log_prob = estimate_log_gaussian_prob(x, means, covariances, cov_type, reg);
        const auto plan = sinkhorn(-log_prob, a, b, sinkhorn_options).plan; // [U, T, K]
        const auto params = m_step(x, plan, cov_type, reg);

        const auto diff = (params.means - means).flatten(1);                    // [U, K*N]
        const auto shift = diff.pow(2).sum(1).sqrt();                           // [U]
        const auto scale = means.flatten(1).pow(2).sum(1).sqrt().clamp_min(1.0e-12);
        const auto newly = torch::logical_and(
            torch::logical_not(converged), (shift / scale) < options.constrained_tol);

        weights = params.weights;
        means = params.means;
        covariances = params.covariances;

        n_iter = torch::where(newly, torch::full_like(n_iter, outer), n_iter);
        converged = torch::logical_or(converged, newly);

        auto cancelled = false;
        if (on_progress) {
            const auto fraction = frac_base
                + frac_span * static_cast<double>(outer) / static_cast<double>(options.constrained_max_iter);
            cancelled = !on_progress(GmmProgress{
                .stage = "constrained",
                .restart = 0,
                .restarts = 1,
                .iter = outer,
                .max_iter = options.constrained_max_iter,
                .lower_bound = 0.0,
                .delta = 0.0,
                .fraction = fraction,
            });
        }
        if (cancelled || converged.all().item<bool>()) {
            break;
        }
    }

    const auto log_prob = estimate_log_gaussian_prob(x, means, covariances, cov_type, reg);
    const auto plan = sinkhorn(-log_prob, a, b, sinkhorn_options).plan;
    const auto step = e_step(x, weights, means, covariances, cov_type, reg);
    return {plan, step.log_prob_norm, converged, n_iter};
}

} // namespace

GmmCovarianceType gmm_covariance_type_for(std::string_view text)
{
    if (text == "full") {
        return GmmCovarianceType::Full;
    }
    if (text == "diag" || text == "diagonal") {
        return GmmCovarianceType::Diagonal;
    }
    throw std::invalid_argument("GaussianMixture covariance_type must be full or diagonal");
}

GmmInitMethod gmm_init_method_for(std::string_view text)
{
    if (text == "kmeans++" || text == "k-means++" || text == "kmeans") {
        return GmmInitMethod::KMeansPlusPlus;
    }
    if (text == "random_from_data") {
        return GmmInitMethod::RandomFromData;
    }
    if (text == "random") {
        return GmmInitMethod::Random;
    }
    throw std::invalid_argument("GaussianMixture init_method must be kmeans++, random_from_data, or random");
}

GaussianMixture::GaussianMixture(GaussianMixtureOptions options)
    : options_(std::move(options))
{
    if (options_.n_components < 1) {
        throw std::invalid_argument("GaussianMixture n_components must be >= 1");
    }
    if (options_.n_init < 1) {
        throw std::invalid_argument("GaussianMixture n_init must be >= 1");
    }
    if (options_.max_iter < 0) {
        throw std::invalid_argument("GaussianMixture max_iter must be >= 0 (0 = init only)");
    }
    if (options_.reg_covar < 0.0) {
        throw std::invalid_argument("GaussianMixture reg_covar must be >= 0");
    }
}

GaussianMixtureFit GaussianMixture::fit(const torch::Tensor& x_in, const GmmProgressCallback& on_progress)
{
    const auto prepared = prepare_input(x_in);
    const auto& x = prepared.x; // [U, T, N]
    const auto u = x.size(0);
    const auto t = x.size(1);
    const auto k = options_.n_components;
    if (t < k) {
        throw std::invalid_argument("GaussianMixture needs at least n_components samples per unit");
    }

    // The plain EM restarts take the first `em_span` of the reported progress; the constrained
    // (Sinkhorn) refinement, when configured, takes the rest. Cancellation is cooperative.
    const auto em_span = options_.target_sizes ? 0.6 : 1.0;
    auto cancelled = false;

    auto generator = torch::make_generator<torch::CPUGeneratorImpl>();
    if (options_.seed) {
        generator.set_current_seed(*options_.seed);
    }

    const auto cov_type = options_.covariance_type;
    const auto reg = options_.reg_covar;

    torch::Tensor best_weights;
    torch::Tensor best_means;
    torch::Tensor best_covariances;
    auto best_lower_bound = torch::full({u}, -std::numeric_limits<double>::infinity(), x.options());
    auto best_converged = torch::zeros({u}, torch::TensorOptions().dtype(torch::kBool));
    auto best_n_iter = torch::zeros({u}, torch::TensorOptions().dtype(torch::kLong));

    for (std::int64_t init = 0; init < options_.n_init; ++init) {
        const auto resp0 = initial_responsibilities(x, k, options_.init_method, generator);
        auto params = m_step(x, resp0, cov_type, reg);

        auto lower_bound = torch::full({u}, -std::numeric_limits<double>::infinity(), x.options());
        auto converged = torch::zeros({u}, torch::TensorOptions().dtype(torch::kBool));
        auto n_iter = torch::full({u}, options_.max_iter, torch::TensorOptions().dtype(torch::kLong));

        for (std::int64_t iteration = 1; iteration <= options_.max_iter; ++iteration) {
            const auto prev_lower_bound = lower_bound;
            const auto step = e_step(x, params.weights, params.means, params.covariances, cov_type, reg);
            lower_bound = step.log_prob_norm.mean(1); // [U]
            params = m_step(x, step.log_resp.exp(), cov_type, reg);

            const auto change = (lower_bound - prev_lower_bound).abs();
            const auto newly = torch::logical_and(torch::logical_not(converged), change < options_.tol);
            n_iter = torch::where(newly, torch::full_like(n_iter, iteration), n_iter);
            converged = torch::logical_or(converged, newly);

            if (on_progress) {
                const auto fraction = em_span
                    * (static_cast<double>(init) + static_cast<double>(iteration) / static_cast<double>(options_.max_iter))
                    / static_cast<double>(options_.n_init);
                cancelled = !on_progress(GmmProgress{
                    .stage = "em",
                    .restart = init,
                    .restarts = options_.n_init,
                    .iter = iteration,
                    .max_iter = options_.max_iter,
                    .lower_bound = lower_bound.mean().item<double>(),
                    .delta = change.mean().item<double>(),
                    .fraction = fraction,
                });
            }
            if (cancelled || (iteration > 1 && converged.all().item<bool>())) {
                break;
            }
        }

        if (options_.max_iter == 0) {
            // Init only: score the initial parameters so n_init selection still works.
            const auto step = e_step(x, params.weights, params.means, params.covariances, cov_type, reg);
            lower_bound = step.log_prob_norm.mean(1);
        }

        const auto better = lower_bound > best_lower_bound; // [U]
        if (init == 0) {
            best_weights = params.weights;
            best_means = params.means;
            best_covariances = params.covariances;
            best_lower_bound = lower_bound;
            best_converged = converged;
            best_n_iter = n_iter;
        } else {
            best_weights = pick_where(better, params.weights, best_weights);
            best_means = pick_where(better, params.means, best_means);
            best_covariances = pick_where(better, params.covariances, best_covariances);
            best_lower_bound = torch::where(better, lower_bound, best_lower_bound);
            best_converged = torch::where(better, converged, best_converged);
            best_n_iter = torch::where(better, n_iter, best_n_iter);
        }

        if (cancelled) {
            break; // caller cancelled during this restart; keep the best fit so far
        }
    }

    weights_ = best_weights.contiguous();
    means_ = best_means.contiguous();
    covariances_ = best_covariances.contiguous();
    fitted_ = true;
    batched_input_ = prepared.batched;

    // Size-constrained refinement (Genevay et al. 2019): the plain EM above is the warm start;
    // now bind the sizes and refine with the Sinkhorn-constrained E-step.
    if (options_.target_sizes && !cancelled) {
        const auto& sizes = *options_.target_sizes;
        if (sizes.size(-1) != k) {
            throw std::invalid_argument("GaussianMixture target_sizes last dimension must equal n_components");
        }
        const auto outcome = run_constrained_em(
            x, weights_, means_, covariances_, sizes, options_, cov_type, reg, on_progress, em_span, 1.0 - em_span);
        weights_ = weights_.contiguous();
        means_ = means_.contiguous();
        covariances_ = covariances_.contiguous();

        GaussianMixtureFit constrained_fit;
        constrained_fit.batched = prepared.batched;
        constrained_fit.weights = drop_unit_axis(weights_, prepared.batched);
        constrained_fit.means = drop_unit_axis(means_, prepared.batched);
        constrained_fit.covariances = drop_unit_axis(covariances_, prepared.batched);
        constrained_fit.responsibilities = drop_unit_axis(outcome.responsibilities, prepared.batched);
        constrained_fit.labels = drop_unit_axis(outcome.responsibilities.argmax(-1), prepared.batched);
        constrained_fit.log_likelihood = drop_unit_axis(outcome.log_prob_norm, prepared.batched);
        constrained_fit.lower_bound = drop_unit_axis(outcome.log_prob_norm.mean(1), prepared.batched);
        constrained_fit.converged = drop_unit_axis(outcome.converged, prepared.batched);
        constrained_fit.n_iter = drop_unit_axis(outcome.n_iter, prepared.batched);
        return constrained_fit;
    }

    // Final E-step with the retained parameters for the reported assignment.
    const auto final_step = e_step(x, weights_, means_, covariances_, cov_type, reg);
    const auto responsibilities = final_step.log_resp.exp();

    GaussianMixtureFit fit;
    fit.batched = prepared.batched;
    fit.weights = drop_unit_axis(weights_, prepared.batched);
    fit.means = drop_unit_axis(means_, prepared.batched);
    fit.covariances = drop_unit_axis(covariances_, prepared.batched);
    fit.responsibilities = drop_unit_axis(responsibilities, prepared.batched);
    fit.labels = drop_unit_axis(responsibilities.argmax(-1), prepared.batched);
    fit.log_likelihood = drop_unit_axis(final_step.log_prob_norm, prepared.batched);
    fit.lower_bound = drop_unit_axis(best_lower_bound, prepared.batched);
    fit.converged = drop_unit_axis(best_converged, prepared.batched);
    fit.n_iter = drop_unit_axis(best_n_iter, prepared.batched);
    return fit;
}

namespace {

[[nodiscard]] EStep apply_fitted(
    const GaussianMixture& model,
    const torch::Tensor& weights,
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& x_in,
    bool& batched_out)
{
    if (!model.fitted()) {
        throw std::logic_error("GaussianMixture: call fit() before predict/predict_proba/score_samples");
    }
    const auto prepared = prepare_input(x_in);
    if (prepared.x.size(0) != weights.size(0)) {
        throw std::invalid_argument("GaussianMixture: input unit axis does not match the fitted model");
    }
    if (prepared.x.size(2) != means.size(2)) {
        throw std::invalid_argument("GaussianMixture: input feature count does not match the fitted model");
    }
    batched_out = prepared.batched;
    return e_step(prepared.x, weights, means, covariances, model.options().covariance_type, model.options().reg_covar);
}

} // namespace

torch::Tensor GaussianMixture::predict(const torch::Tensor& x) const
{
    bool batched = true;
    const auto step = apply_fitted(*this, weights_, means_, covariances_, x, batched);
    return drop_unit_axis(step.log_resp.argmax(-1), batched);
}

torch::Tensor GaussianMixture::predict_proba(const torch::Tensor& x) const
{
    bool batched = true;
    const auto step = apply_fitted(*this, weights_, means_, covariances_, x, batched);
    return drop_unit_axis(step.log_resp.exp(), batched);
}

torch::Tensor GaussianMixture::score_samples(const torch::Tensor& x) const
{
    bool batched = true;
    const auto step = apply_fitted(*this, weights_, means_, covariances_, x, batched);
    return drop_unit_axis(step.log_prob_norm, batched);
}

bool GaussianMixture::fitted() const noexcept
{
    return fitted_;
}

const torch::Tensor& GaussianMixture::weights() const
{
    return weights_;
}

const torch::Tensor& GaussianMixture::means() const
{
    return means_;
}

const torch::Tensor& GaussianMixture::covariances() const
{
    return covariances_;
}

const GaussianMixtureOptions& GaussianMixture::options() const noexcept
{
    return options_;
}

} // namespace leakflow::ml
