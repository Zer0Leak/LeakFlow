#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::ml {

// Ground-truth clustering evaluation (Analyze/Evaluation/Clustering). Joins predicted cluster
// `labels` and true class `truth` (both int `[T]` or `[U, T]`) and emits, on `stats`, the
// Hungarian-reordered (diagonal-aligned) confusion matrix as a float tensor `[U, C, C]` (rows =
// true class, cols = matched cluster), ready for HeatmapPlot. The label-free scores (ARI, NMI,
// purity) and matching-based accuracy are attached as `payload.cluster_stats.*` metadata.
//
// An evaluation-only element: it needs the true class (known key), so it is not part of a real
// blind attack. Pure function of its inputs, so can_replay() is true.
class ClusteringStats final : public Element {
public:
    explicit ClusteringStats(std::string name = "clusterstats0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::ml
