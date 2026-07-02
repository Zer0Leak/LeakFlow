#pragma once

namespace leakflow::plot {

class PipelineControlRuntime;

// Per-frame context handed to every PlotView::draw. It carries the run-level facts
// a view may need but does not own: whether the pipeline is Running (accumulate
// sliders auto-follow only while streaming) and the control runtime (so a view's
// gear button can open its element's control panel). Views ignore the fields they
// do not use (e.g. ScoreView uses neither).
struct PlotDrawContext {
    bool streaming = false;
    PipelineControlRuntime *control_runtime = nullptr;
};

// A self-contained plot type registered with the PlotRuntime. The runtime holds a
// list of PlotViews and, each frame, asks each to draw itself -- it does not know
// whether a view is traces, scores, or a future plot. Each view owns its own copied
// display data, its own UI state, and its own ImGui/ImPlot rendering, so adding a
// new plot type is a new PlotView, not an edit to the shared runtime.
//
// Threading: a view is filled from the pipeline worker thread (its element pushes
// copied snapshot data) and drawn from the UI thread. Each view owns whatever lock
// it needs to serialize those two threads, so its data has no lifetime dependency
// on the PlotRuntime.
class PlotView {
public:
    virtual ~PlotView() = default;

    // Draw all of this view's windows/content into the current ImGui frame. Called
    // on the UI thread; the view locks its own data.
    virtual void draw(const PlotDrawContext &context) = 0;

    // Drop all accumulated display data (a Stop/Start recycle).
    virtual void clear() = 0;

    // Whether the view currently has nothing to show (used for "no plot sessions").
    [[nodiscard]] virtual bool empty() const = 0;
};

} // namespace leakflow::plot
