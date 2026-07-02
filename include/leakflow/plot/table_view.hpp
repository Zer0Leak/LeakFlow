#pragma once

#include "leakflow/plot/plot_runtime.hpp" // TracePlotColor
#include "leakflow/plot/plot_view.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace leakflow::plot {

// Generic table-plot display data (domain-free): a grid of text cells with optional
// per-cell background tint, emphasis, and hover details. A ScoreTablePlot element
// fills this from an attack payload; TableView renders it with ImGui without knowing
// about CPA. Reuses the group concept and always stacks. Each element keeps a bounded
// history of frames (an N-scrubber) so a live run can scrub past updates; offline runs
// simply push one frame -- the final result.
struct TableCell {
    std::string text;
    std::optional<TracePlotColor> tint; // cell background highlight (e.g. correct key)
    bool emphasize = false;             // leading/best cell
    std::vector<std::pair<std::string, std::string>> hover;
};

// One snapshot of the grid at a given N (observation count).
struct TableFrame {
    std::int64_t n = 0;  // observation count / trace number for this frame (>= 1)
    std::string caption; // e.g. "N = 5000"
    std::vector<std::vector<TableCell>> rows;
};

struct TableSnapshot {
    std::uint64_t id = 0;
    std::string group = "default";
    std::string element_name;
    std::string title;
    std::vector<std::string> columns; // headers (first = rank/index, then one per unit)
    std::deque<TableFrame> frames;    // history; back() is the latest
    // -1 => follow the latest frame; otherwise the scrubbed frame index.
    int cursor = -1;
};

// One grid update for element_name. The view finds-or-creates the snapshot, refreshes
// presentation (group/title/columns), appends the frame, and trims history to
// max_history (0 = unbounded, 1 = replace).
struct TableUpdate {
    std::string group;
    std::string title;
    std::vector<std::string> columns;
    TableFrame frame;
    std::size_t max_history = 1;
};

// A self-contained table plot: stacked grids per group window, one line per attack
// unit as columns and ranked guesses as rows, correct-key cells highlighted, with an
// optional per-snapshot history scrubber. It owns its data + UI state + rendering, so
// adding table features never touches the shared PlotRuntime. See
// docs/context/modules/plot.md.
class TableView final : public PlotView {
public:
    // Worker-thread: append a frame to element_name's snapshot (find-or-create).
    void push(std::string element_name, const TableUpdate &update);

    [[nodiscard]] const std::vector<TableSnapshot> &snapshots() const;

    // PlotView:
    void draw(const PlotDrawContext &context) override;
    void clear() override;
    [[nodiscard]] bool empty() const override;

private:
    mutable std::recursive_mutex mutex_;
    std::vector<TableSnapshot> snapshots_;
    std::uint64_t next_id_ = 1;
};

} // namespace leakflow::plot
