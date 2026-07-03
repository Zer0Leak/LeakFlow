#pragma once

#include "leakflow/plot/plot_runtime.hpp" // TracePlot* data types, PlotAxisView
#include "leakflow/plot/plot_view.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow::plot {

// The built-in trace plot as a PlotView: it owns the trace snapshots, all trace UI
// state (per-snapshot slider index / follow-latest, per-group lock + controls-open,
// per-snapshot display state, per-panel axis view), the two-way trace-index / x-axis
// listeners, its own mutex, and the full trace rendering. PlotRuntime creates one
// and registers it (see PlotRuntime::trace_view); every TracePlot element shares it,
// which is how several TracePlots stack/overlay in one group window. Adding another
// plot type is a new PlotView, not an edit here. See docs/design/plotting.md.
class TraceView final : public PlotView {
public:
    // Register or update this element's snapshot. accumulate grows one snapshot with
    // new trace rows (history the slider scrubs); replace updates it in place. Returns
    // the snapshot id. See docs/design/plotting.md and pipeline_controller.md.
    std::uint64_t add_trace(TracePlotSnapshot snapshot);

    // Update only the presentation fields of the snapshot owned by
    // update.element_name (group, labels, layout, context label, center0, x_axis,
    // color, line width, sample rate, order, and the replace-mode trace index),
    // leaving captured data (id, values, shape, annotations, accumulate) untouched.
    // Used by TracePlot to
    // self-apply a ui-control property change with no rerun. No-op if the element has
    // not registered a snapshot yet. force_y_refit clears stored panel y-axis zoom
    // state so explicit controls like center0 apply on the next draw. Returns true
    // when x_axis=time_us had no sample rate and fell back to sample (the caller
    // resets the property).
    bool refresh_trace_display(const TracePlotSnapshot &update, bool force_y_refit = false);

    [[nodiscard]] const std::vector<TracePlotSnapshot> &trace_snapshots() const;

    [[nodiscard]] int &mutable_trace_index(std::uint64_t snapshot_id, int initial_value);
    // Whether an accumulate snapshot's slider auto-follows the newest trace. The
    // renderer sets it false while the user holds the slider and true otherwise.
    [[nodiscard]] bool &mutable_trace_follow_latest(std::uint64_t snapshot_id, bool initial_value);
    // Whether the pipeline is actively streaming (session Running). Set each frame
    // from the PlotDrawContext; accumulate sliders only ride the newest trace while
    // streaming, and stay put once held (Idle) or frozen (Paused) so history can be
    // scrubbed.
    [[nodiscard]] bool streaming() const;
    [[nodiscard]] bool &mutable_group_trace_lock(std::string_view group, bool initial_value);
    [[nodiscard]] TracePlotDisplayState &mutable_trace_display_state(const TracePlotSnapshot &snapshot);
    [[nodiscard]] bool &mutable_group_controls_open(std::string_view group);
    [[nodiscard]] PlotAxisView &mutable_axis_view(std::uint64_t panel_id);
    [[nodiscard]] float &mutable_panel_height(std::string_view panel_key, float default_height);

    // Two-way trace-index sync (Phase 25). When a listener is installed, moving a
    // vertical trace slider notifies it so the owning element's trace_index property
    // can be updated (the session run loop submits a SetProperty).
    void set_trace_index_listener(std::function<void(std::string_view, int)> listener);
    void notify_trace_index(std::string_view element_name, int trace_index);

    // x_axis sync across an overlay sub-group: when one TracePlot's x_axis changes,
    // the other members of its overlay sub-group are notified so the session can set
    // their x_axis to match (an overlay sub-group shares one x-axis).
    void set_x_axis_listener(std::function<void(std::string_view, std::string_view)> listener);
    // Push an x_axis value back to the owning element's property through the listener
    // (the session submits a SetProperty). Used to reconcile a time_us-without-rate
    // fallback so the control panel and graph follow the effective sample axis.
    void notify_x_axis(std::string_view element_name, std::string_view x_axis);

    // PlotView:
    void draw(const PlotDrawContext &context) override;
    void clear() override;
    [[nodiscard]] bool empty() const override;

private:
    mutable std::recursive_mutex mutex_;
    std::vector<TracePlotSnapshot> trace_snapshots_;
    std::uint64_t next_snapshot_id_ = 1;
    bool streaming_ = false;
    std::map<std::uint64_t, int> trace_indices_;
    std::map<std::uint64_t, bool> trace_follow_latest_;
    std::map<std::string, bool> group_trace_locks_;
    std::map<std::uint64_t, TracePlotDisplayState> trace_display_states_;
    std::map<std::string, bool> group_control_windows_;
    std::map<std::uint64_t, PlotAxisView> axis_views_;
    std::map<std::string, float> panel_heights_;
    std::function<void(std::string_view, int)> trace_index_listener_;
    std::function<void(std::string_view, std::string_view)> x_axis_listener_;
};

} // namespace leakflow::plot
