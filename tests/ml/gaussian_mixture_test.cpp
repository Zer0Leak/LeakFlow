#include "leakflow/ml/gaussian_mixture.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

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

template <typename Function>
bool throws_logic_error(Function function)
{
    try {
        function();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

// Well-separated Gaussians so EM has a single obvious optimum. Cluster c of unit u is
// centred at (c * 15 + u) in every dimension with unit variance.
struct Synthetic {
    torch::Tensor x;          // [U, T, N] float32
    torch::Tensor truth;      // [U, T] int64
    torch::Tensor true_means; // [U, K, N] float64
};

Synthetic make_synthetic(std::int64_t u, std::int64_t k, std::int64_t n, std::int64_t per)
{
    torch::manual_seed(1234);
    std::vector<torch::Tensor> unit_x;
    std::vector<torch::Tensor> unit_truth;
    std::vector<torch::Tensor> unit_means;
    for (std::int64_t unit = 0; unit < u; ++unit) {
        std::vector<torch::Tensor> parts;
        std::vector<torch::Tensor> truth_parts;
        std::vector<torch::Tensor> means;
        for (std::int64_t c = 0; c < k; ++c) {
            const auto centre = torch::full({n}, static_cast<double>(c) * 15.0 + static_cast<double>(unit),
                torch::kFloat64);
            means.push_back(centre);
            parts.push_back(centre.to(torch::kFloat32) + torch::randn({per, n}));
            truth_parts.push_back(torch::full({per}, c, torch::kLong));
        }
        unit_x.push_back(torch::cat(parts, 0));
        unit_truth.push_back(torch::cat(truth_parts, 0));
        unit_means.push_back(torch::stack(means, 0));
    }
    return {torch::stack(unit_x, 0), torch::stack(unit_truth, 0), torch::stack(unit_means, 0)};
}

// Every true cluster maps to exactly one predicted label, and that map is a bijection.
bool perfect_clustering(const torch::Tensor& pred_in, const torch::Tensor& truth_in, std::int64_t k)
{
    const auto pred = pred_in.to(torch::kCPU).contiguous();
    const auto truth = truth_in.to(torch::kCPU).contiguous();
    const auto pred_acc = pred.accessor<std::int64_t, 1>();
    const auto truth_acc = truth.accessor<std::int64_t, 1>();
    std::map<std::int64_t, std::int64_t> truth_to_pred;
    for (std::int64_t i = 0; i < pred.size(0); ++i) {
        const auto tt = truth_acc[i];
        const auto pp = pred_acc[i];
        const auto found = truth_to_pred.find(tt);
        if (found == truth_to_pred.end()) {
            truth_to_pred.emplace(tt, pp);
        } else if (found->second != pp) {
            return false;
        }
    }
    std::set<std::int64_t> targets;
    for (const auto& [tt, pp] : truth_to_pred) {
        if (!targets.insert(pp).second) {
            return false;
        }
    }
    return static_cast<std::int64_t>(truth_to_pred.size()) == k;
}

// Each true mean has a recovered mean within tol (permutation-invariant recovery check).
bool means_recovered(const torch::Tensor& recovered, const torch::Tensor& truth, double tol)
{
    const auto distances = torch::cdist(truth.to(torch::kFloat64), recovered.to(torch::kFloat64)); // [K, K]
    const auto nearest = std::get<0>(distances.min(1));
    return nearest.max().item<double>() < tol;
}

} // namespace

int main()
{
    constexpr std::int64_t u = 2;
    constexpr std::int64_t k = 3;
    constexpr std::int64_t n = 4;
    constexpr std::int64_t per = 200;
    constexpr std::int64_t t = k * per;
    const auto data = make_synthetic(u, k, n, per);

    // --- Full covariance, batched over U ---
    leakflow::ml::GaussianMixtureOptions full_options;
    full_options.n_components = k;
    full_options.covariance_type = leakflow::ml::GmmCovarianceType::Full;
    full_options.n_init = 5;
    full_options.seed = 7;
    leakflow::ml::GaussianMixture full_model(full_options);
    const auto full = full_model.fit(data.x);

    if (!expect(full.batched, "full: expected batched output")) {
        return 1;
    }
    if (!expect(full.weights.sizes() == torch::IntArrayRef({u, k}), "full: weights shape wrong")) {
        return 1;
    }
    if (!expect(full.means.sizes() == torch::IntArrayRef({u, k, n}), "full: means shape wrong")) {
        return 1;
    }
    if (!expect(full.covariances.sizes() == torch::IntArrayRef({u, k, n, n}), "full: covariance shape wrong")) {
        return 1;
    }
    if (!expect(full.responsibilities.sizes() == torch::IntArrayRef({u, t, k}), "full: responsibilities shape wrong")) {
        return 1;
    }
    if (!expect(full.converged.all().item<bool>(), "full: not every unit converged")) {
        return 1;
    }
    if (!expect(torch::allclose(full.weights, torch::full({u, k}, 1.0 / k, torch::kFloat64), 0.0, 0.05),
                "full: mixing weights not ~uniform")) {
        return 1;
    }
    for (std::int64_t unit = 0; unit < u; ++unit) {
        if (!expect(means_recovered(full.means[unit], data.true_means[unit], 1.0),
                    "full: means not recovered for a unit")) {
            return 1;
        }
        if (!expect(perfect_clustering(full.labels[unit], data.truth[unit], k),
                    "full: clustering not perfect for a unit")) {
            return 1;
        }
    }

    // predict / predict_proba / score_samples are consistent with the training fit.
    const auto proba = full_model.predict_proba(data.x);
    const auto pred = full_model.predict(data.x);
    const auto scores = full_model.score_samples(data.x);
    if (!expect(torch::allclose(proba.sum(-1), torch::ones({u, t}, torch::kFloat64), 0.0, 1.0e-9),
                "full: predict_proba rows do not sum to 1")) {
        return 1;
    }
    if (!expect(torch::equal(pred, proba.argmax(-1)), "full: predict disagrees with argmax predict_proba")) {
        return 1;
    }
    if (!expect(torch::equal(pred, full.labels), "full: predict disagrees with fitted labels")) {
        return 1;
    }
    if (!expect(torch::allclose(scores, full.log_likelihood, 0.0, 1.0e-9),
                "full: score_samples disagrees with fitted log-likelihood")) {
        return 1;
    }
    if (!expect(torch::isfinite(scores).all().item<bool>(), "full: score_samples not finite")) {
        return 1;
    }

    // --- Diagonal covariance ---
    leakflow::ml::GaussianMixtureOptions diag_options;
    diag_options.n_components = k;
    diag_options.covariance_type = leakflow::ml::GmmCovarianceType::Diagonal;
    diag_options.n_init = 5;
    diag_options.seed = 11;
    leakflow::ml::GaussianMixture diag_model(diag_options);
    const auto diag = diag_model.fit(data.x);
    if (!expect(diag.covariances.sizes() == torch::IntArrayRef({u, k, n}), "diag: covariance shape wrong")) {
        return 1;
    }
    for (std::int64_t unit = 0; unit < u; ++unit) {
        if (!expect(perfect_clustering(diag.labels[unit], data.truth[unit], k),
                    "diag: clustering not perfect for a unit")) {
            return 1;
        }
    }

    // --- Unbatched [T, N] input drops the unit axis throughout ---
    leakflow::ml::GaussianMixture single_model(full_options);
    const auto single = single_model.fit(data.x[0]);
    if (!expect(!single.batched, "single: expected unbatched output")) {
        return 1;
    }
    if (!expect(single.weights.sizes() == torch::IntArrayRef({k}), "single: weights shape wrong")) {
        return 1;
    }
    if (!expect(single.means.sizes() == torch::IntArrayRef({k, n}), "single: means shape wrong")) {
        return 1;
    }
    if (!expect(single.labels.sizes() == torch::IntArrayRef({t}), "single: labels shape wrong")) {
        return 1;
    }
    if (!expect(perfect_clustering(single.labels, data.truth[0], k), "single: clustering not perfect")) {
        return 1;
    }
    if (!expect(single_model.predict(data.x[0]).sizes() == torch::IntArrayRef({t}),
                "single: predict shape wrong")) {
        return 1;
    }

    // --- Error paths ---
    if (!expect(throws_invalid_argument([] {
                    leakflow::ml::GaussianMixtureOptions bad;
                    bad.n_components = 0;
                    leakflow::ml::GaussianMixture model(bad);
                }),
                "expected invalid_argument for n_components < 1")) {
        return 1;
    }
    if (!expect(throws_logic_error([&] {
                    leakflow::ml::GaussianMixture unfit(full_options);
                    (void)unfit.predict(data.x);
                }),
                "expected logic_error for predict before fit")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&] {
                    (void)full_model.score_samples(torch::randn({u, 5, n + 1}));
                }),
                "expected invalid_argument for mismatched feature count")) {
        return 1;
    }

    // --- Size-constrained (Sinkhorn) fit on imbalanced clusters (Genevay 2019) ---
    {
        torch::manual_seed(99);
        constexpr std::int64_t k3 = 3;
        constexpr std::int64_t n3 = 3;
        const std::int64_t sizes[k3] = {700, 200, 100}; // total 1000
        std::vector<torch::Tensor> parts;
        std::vector<torch::Tensor> truth_parts;
        for (std::int64_t c = 0; c < k3; ++c) {
            const auto centre = torch::full({n3}, static_cast<double>(c) * 12.0, torch::kFloat64);
            parts.push_back(centre.to(torch::kFloat32) + torch::randn({sizes[c], n3}));
            truth_parts.push_back(torch::full({sizes[c]}, c, torch::kLong));
        }
        const auto cx = torch::cat(parts, 0);          // [1000, 3]
        const auto ctruth = torch::cat(truth_parts, 0); // [1000]

        leakflow::ml::GaussianMixtureOptions copt;
        copt.n_components = k3;
        copt.covariance_type = leakflow::ml::GmmCovarianceType::Full;
        copt.n_init = 3;
        copt.seed = 5;
        // Scrambled order + proportions (not counts) to exercise binding and rescaling.
        copt.target_sizes = torch::tensor({0.1, 0.7, 0.2}, torch::kFloat64);
        copt.sinkhorn_epsilon = 1.0;
        copt.sinkhorn_max_iter = 2000;
        copt.constrained_max_iter = 50;
        leakflow::ml::GaussianMixture cmodel(copt);
        const auto cfit = cmodel.fit(cx);

        // The constraint is enforced: responsibility column sums match the size multiset.
        const auto col_sums = cfit.responsibilities.sum(0); // [3]
        const auto sorted_cols = std::get<0>(col_sums.sort(/*dim=*/0, /*descending=*/false));
        const auto expected_cols = torch::tensor({100.0, 200.0, 700.0}, torch::kFloat64);
        if (!expect(torch::allclose(sorted_cols, expected_cols, 0.0, 1.0),
                    "constrained: responsibility column sums != target sizes")) {
            return 1;
        }
        // Sizes are correct, so labels still recover the clusters, and the biggest cluster
        // (700 members) receives the biggest quota.
        if (!expect(perfect_clustering(cfit.labels, ctruth, k3), "constrained: clustering not perfect")) {
            return 1;
        }
        if (!expect(std::abs(col_sums.max().item<double>() - 700.0) < 1.0,
                    "constrained: largest quota is not ~700")) {
            return 1;
        }
        // Weights are pinned to sizes / T.
        const auto sorted_w = std::get<0>(cfit.weights.sort(/*dim=*/0, /*descending=*/false));
        const auto expected_w = torch::tensor({0.1, 0.2, 0.7}, torch::kFloat64);
        if (!expect(torch::allclose(sorted_w, expected_w, 0.0, 2.0e-3),
                    "constrained: weights not pinned to sizes/T")) {
            return 1;
        }
    }

    // --- Progress callback: fit reports non-decreasing progress and can cancel early ---
    {
        leakflow::ml::GaussianMixtureOptions options;
        options.n_components = k;
        options.covariance_type = leakflow::ml::GmmCovarianceType::Diagonal;
        options.n_init = 1;
        options.max_iter = 50;
        options.tol = 0.0; // never satisfy convergence, so every iteration reports
        options.seed = 3;

        std::vector<leakflow::ml::GmmProgress> reports;
        leakflow::ml::GaussianMixture model(options);
        const auto fit = model.fit(data.x, [&](const leakflow::ml::GmmProgress& progress) {
            reports.push_back(progress);
            return true;
        });
        if (!expect(!reports.empty(), "progress: callback never called")) {
            return 1;
        }
        if (!expect(reports.front().stage == "em", "progress: first stage should be em")) {
            return 1;
        }
        auto non_decreasing = true;
        for (std::size_t index = 1; index < reports.size(); ++index) {
            if (reports[index].fraction < reports[index - 1].fraction - 1.0e-9) {
                non_decreasing = false;
            }
        }
        if (!expect(non_decreasing, "progress: fraction should be non-decreasing")) {
            return 1;
        }
        if (!expect(reports.back().iter == 50 && reports.back().max_iter == 50,
                    "progress: tol=0 should run all 50 EM iterations")) {
            return 1;
        }
        if (!expect(std::abs(reports.back().fraction - 1.0) < 1.0e-9, "progress: final fraction should be 1.0")) {
            return 1;
        }
        if (!expect(fit.labels.sizes() == torch::IntArrayRef({u, t}), "progress: fit still valid")) {
            return 1;
        }

        // Returning false cancels: the fit stops after the first report but still returns a fit.
        auto calls = 0;
        leakflow::ml::GaussianMixture cancel_model(options);
        const auto cancelled = cancel_model.fit(data.x, [&](const leakflow::ml::GmmProgress&) {
            ++calls;
            return false;
        });
        if (!expect(calls == 1, "progress: cancel should stop after the first report")) {
            return 1;
        }
        if (!expect(cancelled.labels.sizes() == torch::IntArrayRef({u, t}), "progress: cancelled fit still valid")) {
            return 1;
        }

        // The separate cooperative checkpoint also runs inside k-means++ initialization,
        // before EM progress begins. Cancel on its third call to prove initialization has
        // internal boundaries while preserving the numeric best-partial-fit contract.
        auto checkpoint_calls = 0;
        leakflow::ml::GaussianMixture checkpoint_model(options);
        const auto checkpoint_cancelled = checkpoint_model.fit(data.x, {}, [&]() {
            ++checkpoint_calls;
            return checkpoint_calls < 3;
        });
        if (!expect(checkpoint_calls == 3, "checkpoint: initialization did not stop at the requested boundary")) {
            return 1;
        }
        if (!expect(checkpoint_cancelled.labels.sizes() == torch::IntArrayRef({u, t}),
                    "checkpoint: cancelled initialization did not return a valid partial fit")) {
            return 1;
        }
    }

    std::cout << "gaussian_mixture tests passed\n";
    return 0;
}
