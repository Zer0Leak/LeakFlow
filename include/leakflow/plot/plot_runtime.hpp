#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plot {

class PipelineControlRuntime;
class PipelineGraphRuntime;
class PlotView;
class TraceView;

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
    std::string trace_context_label = "trace";
    bool lock_trace_index = false;
    bool center0 = true;
    // When true, add_trace appends this as a distinct snapshot (history overlay)
    // instead of replacing the element's previous snapshot in place. Driven by
    // TracePlot's update_mode (accumulate vs replace). Default replace.
    bool accumulate = false;
    // In accumulate mode, controls annotations independently of the traces: true
    // accumulates per-fold history (a global marker pins to the last row of the fold
    // it arrived with -- the trace the slider follows while streaming -- and per-trace
    // markers keep their own row); false keeps them global (trace_row -1) and replaces
    // the set each buffer, so global markers like PoIs stay visible over the piled-up
    // traces. Driven by TracePlot's annotation_update_mode. Unused in replace mode.
    bool annotations_accumulate = true;
    std::optional<double> sample_rate_hz;
    std::optional<TracePlotColor> color;
    std::vector<std::int64_t> shape;
    std::vector<float> values;
    std::vector<std::int64_t> trace_context_values;
    std::vector<TracePlotAnnotation> annotations;

    [[nodiscard]] std::int64_t rank() const;
    [[nodiscard]] std::int64_t trace_count() const;
    [[nodiscard]] std::int64_t sample_count() const;
    [[nodiscard]] const float *trace_data(std::int64_t trace_index) const;
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
[[nodiscard]] std::pair<double, double> trace_plot_fitted_y_range(double lower, double higher);
[[nodiscard]] std::pair<double, double> trace_plot_centered_y_range(double lower, double higher);

// Persisted per-panel X-axis view (Phase 25). Lets the renderer convert the
// visible window when a panel switches between sample and time_us units so the
// same portion of the trace stays in view.
struct PlotAxisView {
    bool initialized = false;
    bool time_effective = false;
    double x_min = 0.0;
    double x_max = 0.0;
    bool y_fit_initialized = false;
    bool y_user_adjusted = false;
    bool y_fit_centered = false;
    double y_fit_min = 0.0;
    double y_fit_max = 0.0;
};

// A registry of PlotViews plus the shared window/loop entry points. It owns no plot
// data itself: the built-in trace plot lives in a TraceView (created and registered
// at construction, reachable via trace_view()), and every other plot type is another
// registered PlotView (e.g. ScoreView). Drawing/clearing iterate the views without
// knowing their type, so adding a plot type is a new PlotView, not an edit here.
class PlotRuntime {
public:
    PlotRuntime();

    // Register a PlotView (a self-contained plot type, e.g. ScoreView). The runtime
    // draws it each frame without knowing its type. See leakflow/plot/plot_view.hpp.
    void add_view(std::shared_ptr<PlotView> view);
    [[nodiscard]] const std::vector<std::shared_ptr<PlotView>> &views() const;

    // The built-in trace plot, created and registered at construction. TracePlot
    // elements share it; the graph installs the slider/x-axis listeners on it.
    [[nodiscard]] const std::shared_ptr<TraceView> &trace_view() const;

    // True when any registered view has something to show (drives "No plot sessions").
    [[nodiscard]] bool has_sessions() const;

    // Stop/Start recycle: cascade to every registered view's clear().
    void clear();

    [[nodiscard]] std::recursive_mutex &mutex() const;

private:
    std::vector<std::shared_ptr<PlotView>> views_;
    std::shared_ptr<TraceView> trace_view_;
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
