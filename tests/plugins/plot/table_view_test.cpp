#include "leakflow/plot/table_view.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using leakflow::plot::TableCell;
using leakflow::plot::TableFrame;
using leakflow::plot::TableSortDirection;
using leakflow::plot::TableSortSpec;
using leakflow::plot::TableUpdate;
using leakflow::plot::TableUpdateMode;
using leakflow::plot::TableView;

[[nodiscard]] bool expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] TableCell
cell(std::string text,
     std::optional<TableCell::SortValue> sort_value = std::nullopt) {
  TableCell result;
  result.text = std::move(text);
  result.sort_value = std::move(sort_value);
  return result;
}

[[nodiscard]] TableUpdate
update_with_rows(std::int64_t n, std::vector<std::vector<TableCell>> rows,
                 TableUpdateMode mode = TableUpdateMode::AppendFrame,
                 std::size_t max_history = 1) {
  TableUpdate update;
  update.group = "tables";
  update.title = "generic";
  update.columns = {"label", "value"};
  update.frame.n = n;
  update.frame.caption = "N = " + std::to_string(n);
  update.frame.rows = std::move(rows);
  update.max_history = max_history;
  update.update_mode = mode;
  return update;
}

} // namespace

int main() {
  namespace plot = leakflow::plot;

  // Numeric sorting uses typed values rather than formatted text, is stable
  // for ties, and keeps unavailable/NaN rows last in both directions.
  {
    TableFrame frame;
    frame.rows = {
        {cell("ten", std::string("ten")), cell("10", 10.0)},
        {cell("two-a", std::string("two-a")), cell("2", 2.0)},
        {cell("two-b", std::string("two-b")), cell("2.00", 2.0)},
        {cell("missing", std::string("missing")), cell("N/A")},
        {cell("nan", std::string("nan")),
         cell("NaN", std::numeric_limits<double>::quiet_NaN())},
    };
    const auto ascending = plot::table_row_order(
        frame, TableSortSpec{.column_index = 1,
                             .direction = TableSortDirection::Ascending});
    const auto descending = plot::table_row_order(
        frame, TableSortSpec{.column_index = 1,
                             .direction = TableSortDirection::Descending});
    if (!expect(ascending == std::vector<std::size_t>({1, 2, 0, 3, 4}),
                "typed ascending order/stability was wrong") ||
        !expect(descending == std::vector<std::size_t>({0, 1, 2, 3, 4}),
                "typed descending order/missing placement was wrong") ||
        !expect(frame.rows[0][1].text == "10" && frame.rows[1][1].text == "2",
                "sorting should not mutate stored rows")) {
      return 1;
    }

    const auto text_order = plot::table_row_order(
        frame, TableSortSpec{.column_index = 0,
                             .direction = TableSortDirection::Ascending});
    if (!expect(text_order == std::vector<std::size_t>({3, 4, 0, 1, 2}),
                "text ascending order was wrong")) {
      return 1;
    }
  }

  // uint64 keys retain integer ordering beyond double's exact range.
  {
    constexpr auto large = std::uint64_t{9'007'199'254'740'992ULL};
    TableFrame frame;
    frame.rows = {
        {cell("max", std::numeric_limits<std::uint64_t>::max())},
        {cell("large+1", large + 1)},
        {cell("large", large)},
    };
    const auto order = plot::table_row_order(
        frame, TableSortSpec{.column_index = 0,
                             .direction = TableSortDirection::Ascending});
    if (!expect(order == std::vector<std::size_t>({2, 1, 0}),
                "uint64 typed ordering lost precision")) {
      return 1;
    }
  }

  // AppendRows extends the current table in place, ReplaceFrame overwrites it,
  // and legacy AppendFrame independently grows the history scrubber.
  {
    TableView view;
    view.push("comparison",
              update_with_rows(1, {{cell("first"), cell("1", std::int64_t{1})}},
                               TableUpdateMode::AppendFrame, 3));
    view.push("comparison",
              update_with_rows(2,
                               {{cell("second"), cell("2", std::int64_t{2})}},
                               TableUpdateMode::AppendRows, 3));
    auto snapshots = view.snapshots_copy();
    if (!expect(snapshots.size() == 1 && snapshots.front().frames.size() == 1,
                "AppendRows should update the current frame in place") ||
        !expect(snapshots.front().frames.back().rows.size() == 2 &&
                    snapshots.front().frames.back().rows[0][0].text ==
                        "first" &&
                    snapshots.front().frames.back().rows[1][0].text == "second",
                "AppendRows did not create a cumulative latest frame")) {
      return 1;
    }

    view.push(
        "comparison",
        update_with_rows(3, {{cell("replacement"), cell("3", std::int64_t{3})}},
                         TableUpdateMode::ReplaceFrame, 3));
    snapshots = view.snapshots_copy();
    if (!expect(snapshots.front().frames.size() == 1 &&
                    snapshots.front().frames.back().n == 3 &&
                    snapshots.front().frames.back().rows.size() == 1 &&
                    snapshots.front().frames.back().rows[0][0].text ==
                        "replacement",
                "ReplaceFrame should replace the latest frame in place")) {
      return 1;
    }
    view.push("comparison",
              update_with_rows(4,
                               {{cell("latest"), cell("4", std::int64_t{4})}},
                               TableUpdateMode::AppendFrame, 3));
    view.push("comparison",
              update_with_rows(5, {{cell("newer"), cell("5", std::int64_t{5})}},
                               TableUpdateMode::AppendFrame, 3));
    view.push("comparison",
              update_with_rows(6,
                               {{cell("newest"), cell("6", std::int64_t{6})}},
                               TableUpdateMode::AppendFrame, 3));
    snapshots = view.snapshots_copy();
    if (!expect(snapshots.front().frames.size() == 3,
                "bounded history should trim the oldest frame") ||
        !expect(snapshots.front().frames.front().n == 4 &&
                    snapshots.front().frames.back().n == 6,
                "bounded history retained the wrong frames") ||
        !expect(snapshots.front().frames.back().rows.size() == 1 &&
                    snapshots.front().frames.back().rows[0][0].text == "newest",
                "AppendFrame should keep independent frame rows")) {
      return 1;
    }
  }

  // AppendRows unions columns by header, preserving old-column order and
  // backfilling missing cells on both old and incoming rows.
  {
    TableView view;
    view.push("schema", update_with_rows(1, {{cell("first"), cell("1")}}));
    auto expanded = update_with_rows(2, {{cell("2"), cell("extra")}},
                                     TableUpdateMode::AppendRows);
    expanded.columns = {"value", "parameter"};
    view.push("schema", expanded);
    const auto snapshot = view.snapshots_copy().front();
    if (!expect(snapshot.columns ==
                    std::vector<std::string>({"label", "value", "parameter"}),
                "AppendRows column union/order was wrong") ||
        !expect(snapshot.frames.back().rows.size() == 2 &&
                    snapshot.frames.back().rows[0].size() == 3 &&
                    snapshot.frames.back().rows[0][0].text == "first" &&
                    snapshot.frames.back().rows[0][2].text.empty() &&
                    snapshot.frames.back().rows[1][0].text.empty() &&
                    snapshot.frames.back().rows[1][1].text == "2" &&
                    snapshot.frames.back().rows[1][2].text == "extra",
                "AppendRows did not remap/backfill unioned columns")) {
      return 1;
    }

    auto duplicate = update_with_rows(3, {{cell("x"), cell("y")}},
                                      TableUpdateMode::AppendRows);
    duplicate.columns = {"duplicate", "duplicate"};
    bool rejected = false;
    try {
      view.push("schema", duplicate);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!expect(rejected, "AppendRows accepted duplicate column headers")) {
      return 1;
    }
  }

  // Per-element erase leaves other producers intact; global clear removes all
  // snapshots and resets IDs for a Stop/Start recycle.
  {
    TableView view;
    const auto single = update_with_rows(1, {{cell("row"), cell("1")}});
    view.push("a", single);
    view.push("b", single);
    if (!expect(view.update_presentation("b", "comparison", "renamed"),
                "presentation update did not find its producer") ||
        !expect(view.snapshots_copy().back().group == "comparison" &&
                    view.snapshots_copy().back().title == "renamed",
                "presentation update did not apply group/title") ||
        !expect(!view.update_presentation("missing", "x", "y"),
                "presentation update reported a missing producer") ||
        !expect(view.erase("a"),
                "per-snapshot erase did not find its producer") ||
        !expect(!view.erase("missing"), "erase reported a missing producer") ||
        !expect(view.snapshots_copy().size() == 1 &&
                    view.snapshots_copy().front().element_name == "b",
                "per-snapshot erase removed the wrong data")) {
      return 1;
    }
    view.clear();
    if (!expect(view.empty(), "global clear did not empty TableView")) {
      return 1;
    }
    view.push("again", single);
    if (!expect(view.snapshots_copy().front().id == 1,
                "global clear did not reset snapshot IDs")) {
      return 1;
    }
  }

  std::cout << "table_view tests passed\n";
  return 0;
}
