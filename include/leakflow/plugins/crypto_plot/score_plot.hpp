#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/plot/plot_runtime.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace leakflow::plugins::crypto_plot {

// Sink that accumulates AttackStats results into a stacked score plot: one panel
// per selected metric (score + confidence metrics), one line per attack unit, a
// point per streamed buffer at x = observation count. Success is encoded by the
// marker shape (square=success, x=failure, circle=unknown). It reads the crypto
// AttackStatsPayload directly and pushes a generic ScoreSnapshot into the shared
// PlotRuntime, so leakflow_plot stays domain-free and reuses the group windows.
class ScorePlot final : public Element {
  public:
    explicit ScorePlot(std::string name = "scoreplot0");
    ScorePlot(std::shared_ptr<leakflow::plot::PlotRuntime> runtime, std::string name = "scoreplot0");

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;

    [[nodiscard]] std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime() const;
    void set_plot_runtime(std::shared_ptr<leakflow::plot::PlotRuntime> runtime);

  private:
    std::shared_ptr<leakflow::plot::PlotRuntime> runtime_;
    // Fallback x when a buffer carries no attack.observation_count metadata.
    std::int64_t step_ = 0;
};

} // namespace leakflow::plugins::crypto_plot
