#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::ml {

// Batch GMM clustering element (Analyze/Clustering/GMM). Consumes one whole-dataset feature
// tensor ([T, N] or [U, T, N]) and emits per-sample cluster labels ([T] or [U, T], int64),
// wrapping leakflow::ml::GaussianMixture (one GMM per unit). Offline/batch: one buffer in, one
// buffer out. The output is a pure function of the features and properties, so can_replay() is
// true -- tweaking n_components / covariance_type / ... in Idle re-fits from the cached input.
class GaussianMixtureElement final : public Element {
public:
    explicit GaussianMixtureElement(std::string name = "gmm0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::ml
