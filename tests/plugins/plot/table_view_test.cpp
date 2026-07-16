#include "leakflow/plot/table_view.hpp"

#include <algorithm>
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
using leakflow::plot::TableGroupLayout;
using leakflow::plot::TableRowSelector;
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
        !expect(snapshots.front().group_layout == TableGroupLayout::Stacked &&
                    snapshots.front().tab_label.empty(),
                "legacy TableUpdate defaults must remain stacked") ||
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
    if (!expect(view.select_history_frame("comparison", std::size_t{0}),
                "history control could not pin an older frame")) {
      return 1;
    }
    view.push("comparison",
              update_with_rows(
                  7, {{cell("replace-again"), cell("7", std::int64_t{7})}},
                  TableUpdateMode::ReplaceFrame));
    view.push("comparison",
              update_with_rows(
                  8, {{cell("after-replace"), cell("8", std::int64_t{8})}},
                  TableUpdateMode::AppendFrame, 3));
    snapshots = view.snapshots_copy();
    if (!expect(snapshots.front().cursor == -1 &&
                    snapshots.front().frames.back().n == 8,
                "replace left a stale history cursor that stopped following")) {
      return 1;
    }

    TableView trimmed;
    for (std::int64_t n = 1; n <= 3; ++n) {
      trimmed.push("trimmed",
                   update_with_rows(
                       n, {{cell(std::to_string(n)), cell(std::to_string(n))}},
                       TableUpdateMode::AppendFrame, 3));
    }
    if (!expect(trimmed.select_history_frame("trimmed", std::size_t{1}),
                "history control could not pin the middle frame")) {
      return 1;
    }
    trimmed.push("trimmed", update_with_rows(4, {{cell("4"), cell("4")}},
                                             TableUpdateMode::AppendFrame, 3));
    const auto trimmed_snapshot = trimmed.snapshots_copy().front();
    if (!expect(trimmed_snapshot.frames.front().n == 2 &&
                    trimmed_snapshot.cursor == 0,
                "front trimming did not preserve the pinned history frame")) {
      return 1;
    }
  }

  // Tabbed groups retain per-producer identity and presentation state. A
  // stacked producer with the same group remains independently marked for the
  // legacy rendering path.
  {
    TableView view;
    auto stacked = update_with_rows(1, {{cell("stacked"), cell("1")}});
    stacked.group = "shared";
    view.push("stacked", stacked);

    auto first_tab = update_with_rows(1, {{cell("first"), cell("2")}});
    first_tab.group = "shared";
    first_tab.group_layout = TableGroupLayout::Tabs;
    first_tab.tab_label = "First";
    first_tab.tab_order = 20;
    view.push("tab-a", first_tab);

    auto second_tab = update_with_rows(1, {{cell("second"), cell("3")}});
    second_tab.group = "shared";
    second_tab.group_layout = TableGroupLayout::Tabs;
    second_tab.tab_label = "Second";
    second_tab.tab_order = 10;
    view.push("tab-b", second_tab);

    const auto snapshots = view.snapshots_copy();
    const auto snapshot_for = [&](const std::string &element_name)
        -> const leakflow::plot::TableSnapshot * {
      const auto found = std::find_if(
          snapshots.begin(), snapshots.end(), [&](const auto &snapshot) {
            return snapshot.element_name == element_name;
          });
      return found == snapshots.end() ? nullptr : &*found;
    };
    const auto *stacked_snapshot = snapshot_for("stacked");
    const auto *first_snapshot = snapshot_for("tab-a");
    const auto *second_snapshot = snapshot_for("tab-b");
    if (!expect(stacked_snapshot != nullptr && first_snapshot != nullptr &&
                    second_snapshot != nullptr,
                "mixed table layouts lost a producer snapshot") ||
        !expect(stacked_snapshot->group_layout == TableGroupLayout::Stacked &&
                    stacked_snapshot->tab_label.empty(),
                "stacked state was corrupted by a tabbed peer") ||
        !expect(first_snapshot->group_layout == TableGroupLayout::Tabs &&
                    first_snapshot->tab_label == "First" &&
                    first_snapshot->tab_order == 20 &&
                    second_snapshot->group_layout == TableGroupLayout::Tabs &&
                    second_snapshot->tab_label == "Second" &&
                    second_snapshot->tab_order == 10,
                "tabbed group presentation state was not retained") ||
        !expect(view.contains("tab-a") && view.contains("tab-b") &&
                    !view.contains("missing"),
                "producer membership query is wrong")) {
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

  // A row selector is retained as generic presentation metadata. Shared keys
  // let several producer tabs synchronize one choice, while invalid selector
  // schemas fail before a snapshot is created.
  {
    TableView view;
    auto units =
        update_with_rows(1, {{cell("2", std::int64_t{2}), cell("first")},
                             {cell("7", std::int64_t{7}), cell("second")}});
    units.columns = {"Unit", "metric"};
    units.row_selector = TableRowSelector{
        .key = "evaluation.unit",
        .label = "Unit",
        .column = "Unit",
        .values = {{.value = std::int64_t{2}, .label = "2"},
                   {.value = std::int64_t{7}, .label = "7"}},
    };
    view.push("overview", units);
    const auto snapshot = view.snapshots_copy().front();
    if (!expect(snapshot.row_selector.has_value() &&
                    snapshot.row_selector->key == "evaluation.unit" &&
                    snapshot.row_selector->label == "Unit" &&
                    snapshot.row_selector->column == "Unit" &&
                    snapshot.row_selector->values.size() == 2 &&
                    snapshot.frames.back().rows.size() == 2,
                "row selector presentation metadata was not retained")) {
      return 1;
    }
    if (!expect(view.select_row_value("evaluation.unit", std::int64_t{7}) &&
                    view.selected_row_value("evaluation.unit") ==
                        std::optional<TableCell::SortValue>(std::int64_t{7}) &&
                    !plot::table_row_matches_selector(snapshot.frames.back(), 0,
                                                      0, std::int64_t{7}) &&
                    plot::table_row_matches_selector(snapshot.frames.back(), 1,
                                                     0, std::int64_t{7}),
                "typed row selection/filtering did not select only Unit 7")) {
      return 1;
    }

    auto sparse = units;
    sparse.frame.rows.resize(1); // only Unit 2 has data in this tab
    view.push("optional-family", sparse);
    if (!expect(
            view.selected_row_value("evaluation.unit") ==
                    std::optional<TableCell::SortValue>(std::int64_t{7}) &&
                !plot::table_row_matches_selector(
                    view.snapshots_copy().back().frames.back(), 0, 0,
                    std::int64_t{7}),
            "a sparse peer tab changed or fabricated the shared selection")) {
      return 1;
    }

    auto missing_column = units;
    missing_column.row_selector->column = "missing";
    bool rejected_missing_column = false;
    try {
      view.push("invalid-column", missing_column);
    } catch (const std::invalid_argument &) {
      rejected_missing_column = true;
    }
    auto missing_key = units;
    missing_key.row_selector->key.clear();
    bool rejected_missing_key = false;
    try {
      view.push("invalid-key", missing_key);
    } catch (const std::invalid_argument &) {
      rejected_missing_key = true;
    }
    auto duplicate_column = units;
    duplicate_column.columns = {"Unit", "Unit"};
    bool rejected_duplicate_column = false;
    try {
      view.push("duplicate-column", duplicate_column);
    } catch (const std::invalid_argument &) {
      rejected_duplicate_column = true;
    }
    if (!expect(rejected_missing_column && rejected_missing_key &&
                    rejected_duplicate_column &&
                    view.snapshots_copy().size() == 2,
                "invalid row selector schema was accepted or mutated state")) {
      return 1;
    }

    auto changed_key = units;
    changed_key.update_mode = TableUpdateMode::AppendRows;
    changed_key.row_selector->key = "other.unit";
    bool rejected_changed_key = false;
    try {
      view.push("overview", changed_key);
    } catch (const std::invalid_argument &) {
      rejected_changed_key = true;
    }
    if (!expect(rejected_changed_key,
                "AppendRows accepted a changed selector schema") ||
        !expect(view.erase("overview") &&
                    view.selected_row_value("evaluation.unit") ==
                        std::optional<TableCell::SortValue>(std::int64_t{7}) &&
                    view.erase("optional-family") &&
                    !view.selected_row_value("evaluation.unit").has_value(),
                "shared selector state did not survive peers or clear last "
                "owner")) {
      return 1;
    }

    TableView changed_schema;
    auto history = units;
    history.max_history = 3;
    changed_schema.push("history", history);
    history.frame.n = 2;
    changed_schema.push("history", history);
    if (!expect(changed_schema.select_history_frame("history", std::size_t{0}),
                "selector history could not be pinned")) {
      return 1;
    }
    auto replacement = units;
    replacement.update_mode = TableUpdateMode::ReplaceFrame;
    replacement.max_history = 3;
    replacement.row_selector->key = "replacement.unit";
    changed_schema.push("history", replacement);
    const auto replaced_snapshot = changed_schema.snapshots_copy().front();
    if (!expect(
            replaced_snapshot.frames.size() == 1 &&
                replaced_snapshot.cursor == -1 &&
                replaced_snapshot.row_selector->key == "replacement.unit",
            "ReplaceFrame retained history under a changed selector schema")) {
      return 1;
    }

    TableView changed_values;
    changed_values.push("values", units);
    if (!expect(
            changed_values.select_row_value("evaluation.unit", std::int64_t{7}),
            "replacement fixture could not select Unit 7")) {
      return 1;
    }
    auto only_two = units;
    only_two.update_mode = TableUpdateMode::ReplaceFrame;
    only_two.frame.rows.resize(1);
    only_two.row_selector->values.resize(1);
    changed_values.push("values", only_two);
    if (!expect(
            !changed_values.selected_row_value("evaluation.unit").has_value(),
            "replace left a removed selector value observable")) {
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
