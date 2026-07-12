#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <torch/torch.h>

namespace leakflow::ml {

// Gaussian Mixture Model clustering (EM), modelled on scikit-learn's
// sklearn.mixture.GaussianMixture but batched over a leading "unit" axis so many
// independent mixtures fit in one call. Motivating use: MC labeling in blind SCA,
// where the truncated traces are shaped [U, T, N] -- U attack units (e.g. 16 AES key
// bytes, 256 Kyber coefficients), T traces, N PoI samples -- and one GMM is fit per
// unit.
//
// v1 covers the SCA-relevant core: full and diagonal covariance, k-means++ /
// random-from-data init, n_init restarts (best per unit), reg_covar, and
// fit/predict/predict_proba/score_samples. Deferred (extend later): tied/spherical
// covariance, warm_start, caller-provided means/precisions init, sample(), aic/bic.
// All internal math runs in float64 for stability.

enum class GmmCovarianceType {
    Full,     // one N x N covariance per component
    Diagonal, // one length-N variance vector per component
};

enum class GmmInitMethod {
    KMeansPlusPlus,  // k-means++ seeding + a few Lloyd steps (sklearn default flavour)
    RandomFromData,  // means = random distinct data points; equal weights; data covariance
    Random,          // random responsibilities, then one M-step
};

[[nodiscard]] GmmCovarianceType gmm_covariance_type_for(std::string_view text);
[[nodiscard]] GmmInitMethod gmm_init_method_for(std::string_view text);

struct GaussianMixtureOptions {
    std::int64_t n_components = 1;
    GmmCovarianceType covariance_type = GmmCovarianceType::Full;
    double tol = 1.0e-3;        // convergence: stop when the mean log-likelihood gain < tol
    double reg_covar = 1.0e-6;  // added to covariance diagonals for numerical stability
    std::int64_t max_iter = 100;
    std::int64_t n_init = 1;    // restarts; the highest-lower-bound fit is kept per unit
    GmmInitMethod init_method = GmmInitMethod::KMeansPlusPlus;
    std::optional<std::uint64_t> seed; // deterministic init when set

    // Size-constrained (Sinkhorn) fitting -- the GMM realisation of Genevay et al. 2019.
    // When target_sizes is set, fit() first runs the ordinary EM above as a warm start, binds
    // the size multiset to components by mass rank, then refines with a Sinkhorn-constrained
    // E-step (Cuturi 2013) so the assignment respects the known cluster sizes. The mixing
    // weights are then pinned to target_sizes rather than learned. sinkhorn_epsilon dials the
    // assignment from soft (large) to hard (small). target_sizes is a length-K multiset
    // ([K] shared across units, or [U, K]); its order is irrelevant (bound by mass rank) and
    // it may be counts or proportions (rescaled to sum to T internally).
    std::optional<torch::Tensor> target_sizes;
    double sinkhorn_epsilon = 0.05;
    std::int64_t sinkhorn_max_iter = 200;
    std::int64_t constrained_max_iter = 50; // outer constrained-EM iterations
    double constrained_tol = 1.0e-4;        // stop when the relative mean shift < tol
};

// Fitted parameters + the training-data assignment. All tensors are float64 and carry
// the leading U axis (dropped on scalar-unit input to a plain shape). K = n_components.
struct GaussianMixtureFit {
    torch::Tensor weights;          // [U, K]           mixing coefficients (sum to 1 per unit)
    torch::Tensor means;            // [U, K, N]
    torch::Tensor covariances;      // Full: [U, K, N, N]; Diagonal: [U, K, N]
    torch::Tensor responsibilities; // [U, T, K]        predict_proba on the training data
    torch::Tensor labels;           // [U, T] int64     argmax responsibility (predict)
    torch::Tensor log_likelihood;   // [U, T]           per-sample log p(x) (score_samples)
    torch::Tensor lower_bound;      // [U]              final mean log-likelihood
    torch::Tensor converged;        // [U] bool
    torch::Tensor n_iter;           // [U] int64
    bool batched = true;            // false when the caller passed [T, N] (U axis dropped)
};

// Progress of a fit() call, for an optional caller-supplied callback. Framework-agnostic: pure
// values, no LeakFlow types. fit() batches over units, so the meaningful axes are the restart
// (n_init) and the EM / constrained iteration -- not a per-unit index. `fraction` is an overall
// estimate in [0, 1] (fit() weights the EM and Sinkhorn phases). The callback returns false to
// request early cancellation (fit() finishes the current step and returns the best fit so far).
struct GmmProgress {
    std::string_view stage;      // "em" or "constrained"
    std::int64_t restart = 0;    // current restart (0-based); always 0 in the constrained phase
    std::int64_t restarts = 1;   // n_init
    std::int64_t iter = 0;       // current iteration within this phase (1-based)
    std::int64_t max_iter = 0;   // iteration budget for this phase
    double lower_bound = 0.0;    // mean log-likelihood so far (EM phase; 0 in constrained)
    double delta = 0.0;          // last change in lower_bound (EM phase; 0 in constrained)
    double fraction = 0.0;       // overall estimated completion in [0, 1]
};

using GmmProgressCallback = std::function<bool(const GmmProgress&)>;
using GmmCheckpointCallback = std::function<bool()>;

class GaussianMixture {
public:
    explicit GaussianMixture(GaussianMixtureOptions options);

    // Fit per unit. x is [T, N] (single unit) or [U, T, N]. Returns the fit and stores
    // the parameters so predict/predict_proba/score_samples can run on new data. When
    // on_progress is set, it is called after each EM / constrained iteration with a progress
    // estimate; returning false cancels early. on_checkpoint runs before expensive initialization
    // and iteration work so a caller can cooperatively pause or cancel. Cancellation finalizes and
    // returns the best partial fit.
    GaussianMixtureFit fit(const torch::Tensor& x, const GmmProgressCallback& on_progress = {},
        const GmmCheckpointCallback& on_checkpoint = {});

    // Apply the fitted model to new data (same N; [T, N] or [U, T, N]).
    [[nodiscard]] torch::Tensor predict(const torch::Tensor& x) const;       // labels
    [[nodiscard]] torch::Tensor predict_proba(const torch::Tensor& x) const; // responsibilities
    [[nodiscard]] torch::Tensor score_samples(const torch::Tensor& x) const; // per-sample log p(x)

    [[nodiscard]] bool fitted() const noexcept;
    [[nodiscard]] const torch::Tensor& weights() const;     // [U, K]
    [[nodiscard]] const torch::Tensor& means() const;       // [U, K, N]
    [[nodiscard]] const torch::Tensor& covariances() const; // [U,K,N,N] or [U,K,N]
    [[nodiscard]] const GaussianMixtureOptions& options() const noexcept;

private:
    GaussianMixtureOptions options_;
    torch::Tensor weights_;     // [U, K]
    torch::Tensor means_;       // [U, K, N]
    torch::Tensor covariances_; // full: [U,K,N,N]; diag: [U,K,N]
    bool fitted_ = false;
    bool batched_input_ = true;
};

} // namespace leakflow::ml
