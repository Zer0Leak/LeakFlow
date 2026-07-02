#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/plot/score_view.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace leakflow::plugins::crypto_plot {

// Sink that accumulates AttackStats results into a stacked score plot: one panel
// per selected metric (score + confidence metrics), one line per attack unit, a
// point per streamed buffer at x = observation count. Success is encoded by the
// marker shape (square=success, x=failure, circle=unknown). It reads the crypto
// AttackStatsPayload directly and pushes generic points into a ScoreView (a
// self-contained plot registered with the PlotRuntime), so leakflow_plot stays
// domain-free. All ScorePlots sharing a run share one ScoreView (the factory
// creates it), which is how several units/elements stack in one group window.
class ScorePlot final : public Element {
  public:
    explicit ScorePlot(std::string name = "scoreplot0");
    ScorePlot(std::shared_ptr<leakflow::plot::ScoreView> view, std::string name = "scoreplot0");

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;

    [[nodiscard]] std::shared_ptr<leakflow::plot::ScoreView> score_view() const;
    void set_score_view(std::shared_ptr<leakflow::plot::ScoreView> view);

  private:
    void property_changed(std::string_view name) override;

    std::shared_ptr<leakflow::plot::ScoreView> view_;
    // Fallback x when a buffer carries no attack.observation_count metadata.
    std::int64_t step_ = 0;
};

} // namespace leakflow::plugins::crypto_plot
