#pragma once

#include "leakflow/plot/plot_runtime.hpp" // TracePlotColor
#include "leakflow/plot/plot_view.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace leakflow::plot {

// Generic table-plot display data (domain-free): a grid of text cells with
// optional per-cell background tint, emphasis, and hover details. A
// ScoreTablePlot element fills this from an attack payload; TableView renders
// it with ImGui without knowing about CPA. Reuses the group concept and always
// stacks. Each element keeps a bounded history of frames (an N-scrubber) so a
// live run can scrub past updates; offline runs simply push one frame -- the
// final result.
struct TableCell {
  std::string text;
  std::optional<TracePlotColor>
      tint;               // cell background highlight (e.g. correct key)
  bool emphasize = false; // leading/best cell
  std::vector<std::pair<std::string, std::string>> hover;

  // Optional typed value used only for generic column sorting. Keeping it
  // separate from text avoids lexicographic numeric mistakes ("10" before
  // "2") while leaving formatting entirely to the bridge element. Missing
  // values, including floating NaN, sort after available values in either
  // direction.
  using SortValue =
      std::variant<std::int64_t, std::uint64_t, double, std::string>;
  std::optional<SortValue> sort_value;
};

enum class TableSortDirection : std::uint8_t {
  Ascending,
  Descending,
};

struct TableSortSpec {
  std::size_t column_index = 0;
  TableSortDirection direction = TableSortDirection::Ascending;
};

// One snapshot of the grid at a given N (observation count).
struct TableFrame {
  std::int64_t n = 0;  // observation count / trace number for this frame (>= 1)
  std::string caption; // e.g. "N = 5000"
  std::vector<std::vector<TableCell>> rows;
};

// Return a stable display order without mutating the stored rows. A missing
// cell/value and floating NaN sort last for both directions.
[[nodiscard]] std::vector<std::size_t>
table_row_order(const TableFrame &frame, const TableSortSpec &sort);

struct TableSnapshot {
  std::uint64_t id = 0;
  std::string group = "default";
  std::string element_name;
  std::string title;
  std::vector<std::string>
      columns; // headers (first = rank/index, then one per unit)
  std::deque<TableFrame> frames; // history; back() is the latest
  // -1 => follow the latest frame; otherwise the scrubbed frame index.
  int cursor = -1;
  std::optional<TableSortSpec> sort;
};

enum class TableUpdateMode : std::uint8_t {
  // Push an independent history frame. This is the existing/default behavior
  // used by ScoreTablePlot and its scrubber.
  AppendFrame,
  // Replace the latest frame in place (or create it when absent).
  ReplaceFrame,
  // Append incoming rows to the latest frame in place. Column schemas are
  // unioned by header name and missing cells are backfilled.
  AppendRows,
};

// One grid update for element_name. The view finds-or-creates the snapshot,
// refreshes presentation (group/title/columns), appends the frame, and trims
// history to max_history (0 = unbounded, 1 = replace).
struct TableUpdate {
  std::string group;
  std::string title;
  std::vector<std::string> columns;
  TableFrame frame;
  std::size_t max_history = 1;
  TableUpdateMode update_mode = TableUpdateMode::AppendFrame;
};

// A self-contained domain-free table plot: stacked grids per group window with
// typed column sorting, optional per-snapshot history, comparison-row updates,
// and per-producer clearing. Domain bridges decide what rows and columns mean.
// It owns its data + UI state + rendering, so adding table features never
// touches the shared PlotRuntime. See docs/context/modules/plot.md.
class TableView final : public PlotView {
public:
  // Worker-thread: append a frame to element_name's snapshot (find-or-create).
  void push(std::string element_name, const TableUpdate &update);

  // Remove only the snapshot owned by element_name. Unlike clear(), this is
  // safe for one producer to call when multiple elements share the view.
  [[nodiscard]] bool erase(std::string_view element_name);

  // Refresh presentation state without replaying or adding table data.
  [[nodiscard]] bool update_presentation(std::string_view element_name,
                                         std::string group, std::string title);

  [[nodiscard]] const std::vector<TableSnapshot> &snapshots() const;
  [[nodiscard]] std::vector<TableSnapshot> snapshots_copy() const;

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
