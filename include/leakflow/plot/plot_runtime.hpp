#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plot {

class PipelineControlRuntime;
class PipelineGraphRuntime;

inline constexpr auto sample_rate_metadata_key = "capture.sample_rate_hz";

enum class PlotBackend {
    Auto,
    OpenGL3,
    Vulkan,
};

enum class TracePlotLayout {
    Overlay,
    Stacked,
};

enum class TracePlotXAxis {
    Sample,
    TimeUs,
};

// Marker shape for an annotation, used to encode a categorical status independent
// of color/position (e.g. AttackStats success). Circle is the default.
enum class TracePlotAnnotationMarker {
    Circle,
    Square,
    Cross,
};

struct TracePlotColor {
    float red = 0.0F;
    float green = 1.0F;
    float blue = 0.25F;

    [[nodiscard]] bool operator==(const TracePlotColor&) const = default;
};

struct TracePlotDisplayState {
    TracePlotColor color;
    float alpha = 1.0F;
    std::int64_t order = -1;
};

struct TracePlotAnnotation {
    std::int64_t sample_index = 0;
    std::optional<double> value;
    std::optional<double> norm_value;
    std::vector<std::pair<std::string, std::string>> fields;
    std::string label;
    std::string text;
    std::string kind;
    std::optional<std::int64_t> target_index;
    // Trace row this annotation belongs to in an accumulate snapshot, so the
    // renderer shows only the selected trace's annotations (a per-trace history).
    // -1 means it applies to every trace (replace mode / single-trace snapshots).
    std::int64_t trace_row = -1;
    TracePlotAnnotationMarker marker = TracePlotAnnotationMarker::Circle;
};

struct TracePlotSnapshot {
    std::uint64_t id = 0;
    std::string group = "default";
    std::string element_name;
    std::string label;
    std::string title;
    std::string x_label;
    std::string y_label = "leakage";
    TracePlotLayout layout = TracePlotLayout::Overlay;
    TracePlotXAxis x_axis = TracePlotXAxis::Sample;
    double alpha = 1.0;
    double line_width = 1.0;
    std::int64_t initial_trace_index = 0;
    std::int64_t order = -1;
    bool lock_trace_index = false;
    bool center0 = true;
    // When true, add_trace appends this as a distinct snapshot (history overlay)
    // instead of replacing the element's previous snapshot in place. Driven by
    // TracePlot's update_mode (accumulate vs replace). Default replace.
    bool accumulate = false;
    std::optional<double> sample_rate_hz;
    std::optional<TracePlotColor> color;
    std::vector<std::int64_t> shape;
    std::vector<float> values;
    std::vector<TracePlotAnnotation> annotations;

    [[nodiscard]] std::int64_t rank() const;
    [[nodiscard]] std::int64_t trace_count() const;
    [[nodiscard]] std::int64_t sample_count() const;
    [[nodiscard]] const float *trace_data(std::int64_t trace_index) const;
};

// Generic score-plot display data (domain-free): a stacked set of metric panels,
// each with one line series per unit, each series a growing list of (x, y) points
// with a marker shape. A crypto ScorePlot element fills this from an attack payload;
// leakflow_plot renders it without knowing about CPA. Reuses the group concept.
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

// One point to append for a (panel, series) pair at an x. The runtime finds-or-
// creates the panel and series (ordered by first-seen) and appends the point.
struct ScorePointUpdate {
    std::string panel;
    std::string panel_y_label;
    std::string series;
    std::optional<TracePlotColor> color;
    bool secondary = false;
    ScoreSeriesPoint point;
};

struct PlotLoopOptions {
    PlotBackend backend = PlotBackend::Auto;
    std::string window_title = "LeakFlow TracePlot";
    int width = 1280;
    int height = 720;
    // Begin the run immediately on open (--auto-start); otherwise the graph opens in
    // Stopped and waits for the Start button.
    bool auto_start = false;
    // Show the standalone "<window title> Controls" window. Graph runs keep it for
    // player controls and screenshots; plain TracePlot runs can hide it for a
    // cleaner plot-only UI.
    bool show_controls_window = true;
    // Optional per-frame close predicate. Used by non-graph live plotting to close
    // the UI promptly when the background run fails or receives Ctrl+C, while
    // still keeping the window open after normal EOS for inspection.
    std::function<bool()> should_close;
};

[[nodiscard]] std::string_view to_string(PlotBackend backend);
[[nodiscard]] std::string_view to_string(TracePlotLayout layout);
[[nodiscard]] std::string_view to_string(TracePlotXAxis axis);

[[nodiscard]] PlotBackend parse_plot_backend(std::string_view text);
[[nodiscard]] TracePlotLayout parse_trace_plot_layout(std::string_view text);
[[nodiscard]] TracePlotXAxis parse_trace_plot_x_axis(std::string_view text);
// Parse a generic marker hint ("square", "x"/"cross", anything else -> circle).
[[nodiscard]] TracePlotAnnotationMarker parse_trace_plot_annotation_marker(std::string_view text);
[[nodiscard]] double trace_plot_annotation_y_from_norm(double norm_value, double lower, double higher);
[[nodiscard]] std::pair<double, double> trace_plot_centered_y_range(double lower, double higher);

// Persisted per-panel X-axis view (Phase 25). Lets the renderer convert the
// visible window when a panel switches between sample and time_us units so the
// same portion of the trace stays in view.
struct PlotAxisView {
    bool initialized = false;
    bool time_effective = false;
    double x_min = 0.0;
    double x_max = 0.0;
};

class PlotRuntime {
public:
    std::uint64_t add_trace(TracePlotSnapshot snapshot);

    // Update only the presentation fields of the snapshot owned by
    // update.element_name (group, labels, layout, center0, x_axis, color, line
    // width, sample rate, order, and the replace-mode trace index), leaving the
    // captured data (id, values, shape, annotations, accumulate) untouched. Used by
    // TracePlot to self-apply a ui-control property change with no rerun. No-op if
    // the element has not registered a snapshot yet. Returns true when x_axis=time_us
    // had no sample rate and fell back to sample (the caller resets the property).
    bool refresh_trace_display(const TracePlotSnapshot &update);

    // Append points to the score snapshot owned by element_name (find-or-create the
    // snapshot, and each update's panel and series). Used by a ScorePlot element to
    // accumulate attack metrics over streamed buffers. Presentation fields (group,
    // title, x label) are refreshed each call.
    void append_score_points(std::string element_name, std::string group, std::string title, std::string x_label,
                             bool show_secondary, const std::vector<ScorePointUpdate> &updates);
    // Toggle secondary-series display on an existing score snapshot (a ui-control
    // property self-apply), so it flips live without waiting for the next buffer.
    void set_score_show_secondary(std::string_view element_name, bool show_secondary);

    [[nodiscard]] bool has_sessions() const;
    [[nodiscard]] const std::vector<TracePlotSnapshot> &trace_snapshots() const;
    [[nodiscard]] const std::vector<ScoreSnapshot> &score_snapshots() const;

    void clear();

    [[nodiscard]] int &mutable_trace_index(std::uint64_t snapshot_id, int initial_value);
    // Whether an accumulate snapshot's slider auto-follows the newest trace. The
    // renderer sets it false while the user holds the slider and true otherwise.
    [[nodiscard]] bool &mutable_trace_follow_latest(std::uint64_t snapshot_id, bool initial_value);
    // Whether the pipeline is actively streaming (session Running). Accumulate
    // sliders only ride the newest trace while streaming; once a run is held (Idle)
    // or frozen (Paused) the slider stays put so the history can be scrubbed.
    void set_streaming(bool streaming);
    [[nodiscard]] bool streaming() const;
    [[nodiscard]] bool &mutable_group_trace_lock(std::string_view group, bool initial_value);
    [[nodiscard]] TracePlotDisplayState &mutable_trace_display_state(const TracePlotSnapshot &snapshot);
    [[nodiscard]] bool &mutable_group_controls_open(std::string_view group);
    [[nodiscard]] PlotAxisView &mutable_axis_view(std::uint64_t panel_id);
    // User-draggable pixel height for a stacked score panel, keyed by
    // "<snapshotId>/<metric>". The renderer resizes it via a splitter handle.
    [[nodiscard]] float &mutable_score_panel_height(const std::string &key, float initial_value);

    // Two-way trace-index sync (Phase 25). When a listener is installed, moving a
    // vertical trace slider notifies it so the owning element's trace_index
    // property can be updated (the session run loop submits a SetProperty).
    void set_trace_index_listener(std::function<void(std::string_view, int)> listener);
    void notify_trace_index(std::string_view element_name, int trace_index);

    // x_axis sync across an overlay sub-group: when one TracePlot's x_axis changes,
    // the other members of its overlay sub-group are notified so the session can
    // set their x_axis to match (an overlay sub-group shares one x-axis).
    void set_x_axis_listener(std::function<void(std::string_view, std::string_view)> listener);
    // Push an x_axis value back to the owning element's property through the listener
    // (the session submits a SetProperty). Used to reconcile a time_us-without-rate
    // fallback so the control panel and graph follow the effective sample axis.
    void notify_x_axis(std::string_view element_name, std::string_view x_axis);

    [[nodiscard]] std::recursive_mutex &mutex() const;

private:
    std::vector<TracePlotSnapshot> trace_snapshots_;
    std::vector<ScoreSnapshot> score_snapshots_;
    std::uint64_t next_snapshot_id_ = 1;
    bool streaming_ = false;
    std::map<std::uint64_t, int> trace_indices_;
    std::map<std::uint64_t, bool> trace_follow_latest_;
    std::map<std::string, bool> group_trace_locks_;
    std::map<std::uint64_t, TracePlotDisplayState> trace_display_states_;
    std::map<std::string, bool> group_control_windows_;
    std::map<std::uint64_t, PlotAxisView> axis_views_;
    std::map<std::string, float> score_panel_heights_;
    std::function<void(std::string_view, int)> trace_index_listener_;
    std::function<void(std::string_view, std::string_view)> x_axis_listener_;
    mutable std::recursive_mutex mutex_;
};

void draw_plot_runtime(PlotRuntime &runtime, PipelineControlRuntime *control_runtime = nullptr);
void run_until_closed(PlotRuntime &runtime, PipelineGraphRuntime &graph_runtime, const PlotLoopOptions &options = {});
void run_until_closed(PlotRuntime &runtime, PipelineGraphRuntime &graph_runtime,
                      PipelineControlRuntime &control_runtime, const PlotLoopOptions &options = {});
void run_until_closed(PlotRuntime &runtime, PipelineControlRuntime &control_runtime,
                      const PlotLoopOptions &options = {});
void run_until_closed(PlotRuntime &runtime, const PlotLoopOptions &options = {});

} // namespace leakflow::plot
