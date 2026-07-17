#pragma once

#include "leakflow/plot/plot_runtime.hpp" // TracePlotColor
#include "leakflow/plot/plot_view.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
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
// it with ImGui without knowing about CPA. Groups can use legacy stacked tables
// or explicitly ordered tabs. Each element keeps a bounded history of frames
// (an N-scrubber) so a live run can scrub past updates; offline runs simply
// push one frame -- the final result.
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

struct TableRowSelectorValue {
  TableCell::SortValue value;
  std::string label;
};

// Optional view-local row selector. The generic table does not interpret the
// selected values: the bridge supplies ordered typed choices and the named
// column carrying each row's typed value. Snapshots that use the same non-empty
// key share one selection (for example, a Unit selector across several tabs
// owned by one element).
struct TableRowSelector {
  std::string key;
  std::string label;
  std::string column;
  std::vector<TableRowSelectorValue> values;
};

enum class TableHeatmapValueFormat : std::uint8_t {
  Number,
  // The stored value is a unit fraction and is displayed as a percentage.
  Percentage,
};

// Optional matrix page rendered inside a tabbed table snapshot. The generic
// view treats selector_value as an opaque typed page key; a domain bridge owns
// the matrix meaning, labels, ordering, and any presentation-only
// normalization applied before the page is copied here. Pages in one frame may
// have different rectangular shapes.
struct TableHeatmapPage {
  TableCell::SortValue selector_value;
  std::string caption;
  std::string row_axis_label = "row";
  std::string col_axis_label = "column";
  std::vector<std::string> row_labels;
  std::vector<std::string> col_labels;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::vector<double> data;
  double vmin = 0.0;
  double vmax = 1.0;
  // A non-empty reason represents a deliberately unavailable page. Such a
  // page carries no matrix allocation but stays selectable and visible.
  std::string unavailable_reason;
  // Optional bridge-supplied presentation text. TableView remains unaware of
  // the matrix domain while still giving every cell a useful hover tooltip.
  std::string value_label = "value";
  TableHeatmapValueFormat value_format = TableHeatmapValueFormat::Number;
  // Optional dense support matrix for numeric cell-hover metrics. When
  // present, counts has the same row-major shape as data and count_total is
  // their sum. The bridge chooses the domain-neutral support label.
  std::vector<std::uint64_t> counts;
  std::uint64_t count_total = 0;
  std::string count_label = "count";
};

struct TableHeatmapFrame {
  std::vector<TableHeatmapPage> pages;
};

enum class TableGroupLayout : std::uint8_t {
  // Preserve the existing behavior: every producer is drawn as a vertically
  // stacked table in its group window.
  Stacked,
  // Draw every producer in the group as an independent tab. Each tab retains
  // its own table state (history cursor, sorting, and per-table clear).
  Tabs,
};

// One snapshot of the grid at a given N (observation count).
struct TableFrame {
  std::int64_t n = 0;  // observation count / trace number for this frame (>= 1)
  std::string caption; // e.g. "N = 5000"
  std::vector<std::vector<TableCell>> rows;
  // Appended to preserve positional aggregate initialization of table frames.
  // When present, the frame renders the selected matrix page instead of rows;
  // rows still carry the selector facet values used by TableRowSelector.
  std::optional<TableHeatmapFrame> heatmap;
};

// ImPlot's row-major PlotHeatmap places input row zero nearest the maximum Y
// bound. Use this coordinate for row-label ticks so labels and matrix rows stay
// paired without inverting the plot axis. Throws when row_index is outside the
// declared row count.
[[nodiscard]] double table_heatmap_row_axis_position(std::size_t row_index,
                                                     std::size_t row_count);

struct TableHeatmapCellIndex {
  std::size_t row = 0;
  std::size_t column = 0;

  [[nodiscard]] bool
  operator==(const TableHeatmapCellIndex &) const noexcept = default;
};

struct TableHeatmapCellMetrics {
  std::uint64_t cell_support = 0;
  std::uint64_t row_support = 0;
  std::uint64_t column_support = 0;
  std::uint64_t total_support = 0;

  [[nodiscard]] bool
  operator==(const TableHeatmapCellMetrics &) const noexcept = default;
};

// Resolve ImPlot coordinates to the row-major cell under the mouse. Heatmap
// row zero is displayed at the maximum Y bound, so the returned row index is
// reversed relative to floor(y). Bounds are half-open: [0, cols) x [0, rows).
[[nodiscard]] std::optional<TableHeatmapCellIndex>
table_heatmap_cell_at(double x, double y, std::size_t row_count,
                      std::size_t column_count);

// Derive local support metrics for a cell when the optional dense count matrix
// is available. Returns no value for an invalid page/cell or absent counts.
[[nodiscard]] std::optional<TableHeatmapCellMetrics>
table_heatmap_cell_metrics(const TableHeatmapPage &page,
                           const TableHeatmapCellIndex &cell);

// Return a stable display order without mutating the stored rows. A missing
// cell/value and floating NaN sort last for both directions.
[[nodiscard]] std::vector<std::size_t>
table_row_order(const TableFrame &frame, const TableSortSpec &sort);

// The pure predicate used by row-selector rendering. Exposed so bridges can
// verify selector/filter contracts without requiring an ImGui interaction.
[[nodiscard]] bool
table_row_matches_selector(const TableFrame &frame, std::size_t row_index,
                           std::size_t column_index,
                           const TableCell::SortValue &selected_value);

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
  // Appended to preserve positional aggregate initialization of the legacy
  // snapshot fields.
  TableGroupLayout group_layout = TableGroupLayout::Stacked;
  std::string tab_label;
  // Lower values render first within a tab group. Equal values preserve
  // producer insertion order.
  int tab_order = 0;
  std::optional<TableRowSelector> row_selector;
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
  // Appended so legacy positional aggregate initialization stays valid.
  // Defaults preserve every existing producer's stacked rendering.
  TableGroupLayout group_layout = TableGroupLayout::Stacked;
  // Visible label when group_layout is Tabs. An empty label falls back to the
  // producer's title and then its element name.
  std::string tab_label;
  int tab_order = 0;
  // A domain-neutral, view-local row facet. It changes only which rows are
  // drawn; the retained table data and pipeline payload stay untouched.
  std::optional<TableRowSelector> row_selector;
};

// A self-contained domain-free table plot: stacked grids or independent tabs
// per group window, with typed column sorting, optional per-snapshot history,
// comparison-row updates, and per-producer clearing. Domain bridges decide what
// rows and columns mean. It owns its data + UI state + rendering, so adding
// table features never touches the shared PlotRuntime. See
// docs/context/modules/plot.md.
class TableView final : public PlotView {
public:
  // Worker-thread: append a frame to element_name's snapshot (find-or-create).
  void push(std::string element_name, const TableUpdate &update);

  // Remove only the snapshot owned by element_name. Unlike clear(), this is
  // safe for one producer to call when multiple elements share the view.
  [[nodiscard]] bool erase(std::string_view element_name);

  // Query producer membership without copying retained table rows or history.
  [[nodiscard]] bool contains(std::string_view element_name) const;

  // Refresh presentation state without replaying or adding table data.
  [[nodiscard]] bool update_presentation(std::string_view element_name,
                                         std::string group, std::string title);

  // View-local controls used by the ImGui widgets and headless controllers.
  // A selected row value is shared by every snapshot with selector_key.
  [[nodiscard]] bool select_row_value(std::string_view selector_key,
                                      TableCell::SortValue value);
  [[nodiscard]] std::optional<TableCell::SortValue>
  selected_row_value(std::string_view selector_key) const;
  // nullopt (or the latest frame index) follows latest; an older valid index
  // pins the history scrubber to that frame.
  [[nodiscard]] bool
  select_history_frame(std::string_view element_name,
                       std::optional<std::size_t> frame_index);

  [[nodiscard]] const std::vector<TableSnapshot> &snapshots() const;
  [[nodiscard]] std::vector<TableSnapshot> snapshots_copy() const;

  // PlotView:
  void draw(const PlotDrawContext &context) override;
  void clear() override;
  [[nodiscard]] bool empty() const override;

private:
  mutable std::recursive_mutex mutex_;
  std::vector<TableSnapshot> snapshots_;
  std::map<std::string, TableCell::SortValue> selected_row_values_;
  std::uint64_t next_id_ = 1;
};

} // namespace leakflow::plot
