#pragma once

#include "leakflow/plot/plot_view.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace leakflow::plot {

// Generic matrix-heatmap display data (domain-free): a stack of U matrices per producing
// element, drawn as an ImPlot heatmap with a sequential colormap and colorbar, plus a slider
// to pick which unit is shown. A HeatmapPlot element fills this; HeatmapView renders it without
// knowing the matrices are per-byte confusion matrices.
struct HeatmapSnapshot {
    std::uint64_t id = 0;
    std::string element_name; // producer; new matrices from the same element replace in place
    std::string title;
    std::string row_label = "row";
    std::string col_label = "col";
    std::string value_label;               // shown above the plot, e.g. "row-normalised"
    std::vector<std::string> unit_captions; // optional per-unit caption (e.g. "byte 0  acc=0.09")
    std::int64_t units = 1;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<double> data; // units * rows * cols, row-major per unit
    double vmin = 0.0;
    double vmax = 1.0;
};

// A self-contained matrix-heatmap plot (one window per producing element). Owns its copied
// display data, UI state, and rendering, so adding it never touches the shared PlotRuntime.
class HeatmapView final : public PlotView {
public:
    HeatmapView() = default;

    // Worker-thread: replace the U-matrix stack owned by element_name (find-or-create the
    // snapshot). data is units * rows * cols, row-major per unit.
    void set_matrix(
        std::string element_name,
        std::string title,
        std::string row_label,
        std::string col_label,
        std::string value_label,
        std::vector<std::string> unit_captions,
        std::int64_t units,
        std::int64_t rows,
        std::int64_t cols,
        std::vector<double> data,
        double vmin,
        double vmax);

    [[nodiscard]] const std::vector<HeatmapSnapshot>& snapshots() const;

    // PlotView:
    void draw(const PlotDrawContext& context) override;
    void clear() override;
    [[nodiscard]] bool empty() const override;

private:
    mutable std::recursive_mutex mutex_;
    std::vector<HeatmapSnapshot> snapshots_;
    // Selected unit per producing element, preserved across live updates (like ScoreView's
    // panel heights) so the slider stays put while data streams in.
    std::map<std::string, int> selected_unit_;
    std::uint64_t next_id_ = 1;
};

} // namespace leakflow::plot
