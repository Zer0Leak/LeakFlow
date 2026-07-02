#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/plot/trace_view.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace leakflow::plot {
class PlotRuntime;
}

namespace leakflow::plugins::plot {

// Sink that renders traces (and annotations) into a TraceView. Several TracePlots
// sharing a TraceView stack/overlay in one group window; the TraceView is the shared
// plot registered with the PlotRuntime (see leakflow/plot/trace_view.hpp).
class TracePlot final : public Element {
  public:
    explicit TracePlot(std::string name = "traceplot0");
    TracePlot(std::shared_ptr<leakflow::plot::TraceView> view, std::string name = "traceplot0");
    // Convenience: fill the runtime's built-in trace view (keeps app/tutorial code
    // that only has a PlotRuntime working).
    TracePlot(const std::shared_ptr<leakflow::plot::PlotRuntime> &runtime, std::string name = "traceplot0");

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;

    [[nodiscard]] std::shared_ptr<leakflow::plot::TraceView> trace_view() const;
    void set_trace_view(std::shared_ptr<leakflow::plot::TraceView> view);

  private:
    // Resolve update_mode (auto/accumulate/replace) into the read-only
    // active_update_mode and return whether snapshots should accumulate. auto
    // follows liveness: live-driven overlays history, one-run replaces in place.
    bool resolve_accumulate();
    void update_active_update_mode();
    // Apply ui-control (presentation) property changes to this element's registered
    // snapshot directly, so they show immediately in any state without a rerun.
    void refresh_display();
    void property_changed(std::string_view name) override;
    void live_driven_changed() override;

    std::shared_ptr<leakflow::plot::TraceView> view_;
    // Warn only once per run when x_axis=time_us has no sample rate; reset in start().
    bool warned_time_us_no_rate_ = false;
};

} // namespace leakflow::plugins::plot
