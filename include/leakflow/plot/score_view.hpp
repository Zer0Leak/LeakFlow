#pragma once

#include "leakflow/plot/plot_runtime.hpp" // TracePlotColor, TracePlotAnnotationMarker
#include "leakflow/plot/plot_view.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plot {

// Generic score-plot display data (domain-free): a stacked set of metric panels,
// each with one line series per unit, each series a growing list of (x, y) points
// with a marker shape. A ScorePlot element fills this from an attack payload;
// ScoreView renders it without knowing about CPA. Reuses the group concept.
struct ScoreSeriesPoint {
    double x = 0.0;
    double y = 0.0;
    TracePlotAnnotationMarker marker = TracePlotAnnotationMarker::Circle;
    std::vector<std::pair<std::string, std::string>> fields; // hover details
};

struct ScoreSeries {
    std::string label; // e.g. "byte 03"
    std::optional<TracePlotColor> color;
    // A secondary series shares its label/color with the primary (so one legend entry
    // toggles both) and is drawn fainter without markers. Gated by show_secondary.
    bool secondary = false;
    std::vector<ScoreSeriesPoint> points;
};

struct ScorePanel {
    std::string metric; // panel key, e.g. "score" / "relative_margin"
    std::string y_label;
    std::vector<ScoreSeries> series; // one per unit (plus optional secondary series)
};

struct ScoreSnapshot {
    std::uint64_t id = 0;
    std::string group = "default";
    std::string element_name;
    std::string title;
    std::string x_label = "traces (N)";
    // Whether secondary series (e.g. the second-best score) are drawn. Producers
    // still accumulate them, so this can be toggled live without losing history.
    bool show_secondary = false;
    std::vector<ScorePanel> panels; // always stacked
};

// One point to append for a (panel, series) pair at an x. The view finds-or-creates
// the panel and series (ordered by first-seen) and appends the point.
struct ScorePointUpdate {
    std::string panel;
    std::string panel_y_label;
    std::string series;
    std::optional<TracePlotColor> color;
    bool secondary = false;
    ScoreSeriesPoint point;
};

// A self-contained score plot: stacked metric panels per group window, one line per
// unit, markers by success, per-unit "latest wrong" vertical lines, draggable panel
// heights. It owns all its data + UI state + rendering, so adding score features
// never touches the shared PlotRuntime. See docs/context/modules/plot.md.
class ScoreView final : public PlotView {
public:
    ScoreView() = default;

    // Worker-thread: append points to the snapshot owned by element_name (find-or-
    // create the snapshot, and each update's panel and series). Presentation fields
    // (group, title, x label, show_secondary) are refreshed each call.
    void append_points(std::string element_name, std::string group, std::string title, std::string x_label,
                       bool show_secondary, const std::vector<ScorePointUpdate> &updates);
    // Toggle secondary-series display on an existing snapshot (a ui-control property
    // self-apply), so it flips live without waiting for the next buffer.
    void set_show_secondary(std::string_view element_name, bool show_secondary);

    [[nodiscard]] const std::vector<ScoreSnapshot> &snapshots() const;

    // PlotView:
    void draw(const PlotDrawContext &context) override;
    void clear() override;
    [[nodiscard]] bool empty() const override;

private:
    // Owned so the view is fully self-contained: its data has no lifetime or locking
    // dependency on the PlotRuntime (the worker append and UI draw serialize here).
    mutable std::recursive_mutex mutex_;
    std::vector<ScoreSnapshot> snapshots_;
    std::uint64_t next_id_ = 1;
    // User-draggable panel heights, keyed by "<snapshotId>/<ordinal>/<metric>".
    std::map<std::string, float> panel_heights_;
};

} // namespace leakflow::plot
