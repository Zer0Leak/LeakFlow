#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <memory>
#include <optional>
#include <string>

namespace leakflow::plot {
class HeatmapView;
}

namespace leakflow::plugins::plot {

// Sink that renders a matrix tensor as an ImPlot heatmap (Plot/Heatmap). Generic: any `[R, C]`
// or `[U, R, C]` tensor on `matrix`; a batched tensor is aggregated (sum over U, or a selected
// `unit`). `normalize` (none/row/col) picks the color scale -- row-normalisation turns a
// ClusteringStats confusion into a per-class recall heatmap. Domain-free; knows nothing about
// clustering.
class HeatmapPlot final : public Element {
public:
    explicit HeatmapPlot(std::string name = "heatmap0");
    HeatmapPlot(std::shared_ptr<leakflow::plot::HeatmapView> view, std::string name = "heatmap0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;

    [[nodiscard]] std::shared_ptr<leakflow::plot::HeatmapView> heatmap_view() const;

private:
    std::shared_ptr<leakflow::plot::HeatmapView> view_;
};

} // namespace leakflow::plugins::plot
