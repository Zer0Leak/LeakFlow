#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/plot/table_view.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace leakflow::plugins::crypto_plot {

// Sink that renders AttackStats results as a live scoreboard table: columns are the
// attack units, rows are the ranked candidate guesses (each cell = guess + score),
// sorted by score (default) or by guess, with the correct-key cell highlighted. It
// reads the crypto AttackStatsPayload directly and pushes a generic table frame into
// a TableView, so leakflow_plot stays domain-free. Works the same live (a frame per
// streamed buffer, kept up to max_history) and offline (one frame = the final result).
// All ScoreTablePlots in a run share one TableView (the factory creates it).
class ScoreTablePlot final : public Element {
  public:
    explicit ScoreTablePlot(std::string name = "scoretableplot0");
    ScoreTablePlot(std::shared_ptr<leakflow::plot::TableView> view, std::string name = "scoretableplot0");

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;

    [[nodiscard]] std::shared_ptr<leakflow::plot::TableView> table_view() const;
    void set_table_view(std::shared_ptr<leakflow::plot::TableView> view);

  private:
    std::shared_ptr<leakflow::plot::TableView> view_;
    // Fallback x when a buffer carries no attack.observation_count metadata.
    std::int64_t step_ = 0;
};

} // namespace leakflow::plugins::crypto_plot
