#include "leakflow/plot/table_view.hpp"

#include "plot_render_util.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace leakflow::plot {
namespace {

[[nodiscard]] bool sort_value_missing(const TableCell::SortValue &value) {
  if (const auto *number = std::get_if<double>(&value)) {
    return std::isnan(*number);
  }
  return false;
}

[[nodiscard]] bool sort_value_numeric(const TableCell::SortValue &value) {
  return !std::holds_alternative<std::string>(value);
}

[[nodiscard]] long double
numeric_sort_value(const TableCell::SortValue &value) {
  return std::visit(
      [](const auto &typed) -> long double {
        using Value = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Value, std::string>) {
          return 0.0L;
        } else {
          return static_cast<long double>(typed);
        }
      },
      value);
}

// Three-way comparison for available values. Bridges should use one value type
// consistently within a column; mixed numeric alternatives still compare by
// numeric value, while text sorts before numeric data deterministically.
[[nodiscard]] int compare_sort_values(const TableCell::SortValue &left,
                                      const TableCell::SortValue &right) {
  if (left.index() == right.index()) {
    return std::visit(
        [](const auto &left_value, const auto &right_value) -> int {
          using Left = std::decay_t<decltype(left_value)>;
          using Right = std::decay_t<decltype(right_value)>;
          if constexpr (!std::is_same_v<Left, Right>) {
            return 0;
          } else if (left_value < right_value) {
            return -1;
          } else if (right_value < left_value) {
            return 1;
          } else {
            return 0;
          }
        },
        left, right);
  }

  const auto left_numeric = sort_value_numeric(left);
  const auto right_numeric = sort_value_numeric(right);
  if (left_numeric && right_numeric) {
    const auto left_value = numeric_sort_value(left);
    const auto right_value = numeric_sort_value(right);
    return left_value < right_value ? -1 : (right_value < left_value ? 1 : 0);
  }
  if (left_numeric != right_numeric) {
    return left_numeric ? 1 : -1;
  }
  return left.index() < right.index() ? -1 : 1;
}

[[nodiscard]] const TableCell::SortValue *
sort_value_at(const std::vector<TableCell> &row, std::size_t column) {
  if (column >= row.size() || !row[column].sort_value.has_value()) {
    return nullptr;
  }
  return &*row[column].sort_value;
}

[[nodiscard]] bool column_is_sortable(const TableFrame &frame,
                                      std::size_t column) {
  return std::ranges::any_of(frame.rows, [column](const auto &row) {
    return column < row.size() && row[column].sort_value.has_value();
  });
}

[[nodiscard]] std::map<std::string, std::size_t>
column_indexes(const std::vector<std::string> &columns,
               std::string_view operation) {
  std::map<std::string, std::size_t> result;
  for (std::size_t index = 0; index < columns.size(); ++index) {
    if (!result.emplace(columns[index], index).second) {
      throw std::invalid_argument("TableView " + std::string(operation) +
                                  " requires unique column headers");
    }
  }
  return result;
}

[[nodiscard]] TableFrame
remap_frame_columns(const TableFrame &frame,
                    const std::vector<std::string> &source_columns,
                    const std::vector<std::string> &target_columns) {
  const auto target_indexes = column_indexes(target_columns, "column remap");
  TableFrame result;
  result.n = frame.n;
  result.caption = frame.caption;
  result.rows.reserve(frame.rows.size());
  for (const auto &source_row : frame.rows) {
    std::vector<TableCell> target_row(target_columns.size());
    const auto cell_count = std::min(source_row.size(), source_columns.size());
    for (std::size_t source_index = 0; source_index < cell_count;
         ++source_index) {
      target_row[target_indexes.at(source_columns[source_index])] =
          source_row[source_index];
    }
    result.rows.push_back(std::move(target_row));
  }
  return result;
}

// Center a cell's/header's text within the current table column (ImGui aligns
// left by default). Leaves the drawn text as the "last item" so IsItemHovered
// still works.
void draw_centered_text(const char *text, const ImVec4 &color) {
  const auto text_width = ImGui::CalcTextSize(text).x;
  const auto avail = ImGui::GetContentRegionAvail().x;
  if (avail > text_width) {
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - text_width) * 0.5F);
  }
  ImGui::TextColored(color, "%s", text);
}

[[nodiscard]] bool draw_table_snapshot(TableSnapshot &snapshot,
                                       float height_budget) {
  if (snapshot.frames.empty() || snapshot.columns.empty()) {
    return false;
  }
  const auto content_top = ImGui::GetCursorPosY();
  ImGui::PushID(static_cast<int>(snapshot.id));
  if (!snapshot.title.empty()) {
    ImGui::SeparatorText(snapshot.title.c_str());
  }
  if (ImGui::SmallButton("Clear")) {
    ImGui::PopID();
    return true;
  }

  // History cursor: -1 follows the latest frame; otherwise the pinned frame
  // index.
  const auto frame_count = static_cast<int>(snapshot.frames.size());
  const auto frame_index =
      (snapshot.cursor < 0 || snapshot.cursor >= frame_count) ? frame_count - 1
                                                              : snapshot.cursor;
  const auto &frame = snapshot.frames[static_cast<std::size_t>(frame_index)];

  const auto column_count = static_cast<int>(snapshot.columns.size());
  const auto &style = ImGui::GetStyle();

  // Auto-size each column to the widest of its header + cells (so nothing clips
  // and no manual resize is needed), then size the table to that content -- a
  // window wider than the table then just leaves plain background, not an empty
  // bordered column. ScrollX still kicks in once the content is wider than the
  // region.
  std::vector<float> column_widths(static_cast<std::size_t>(column_count),
                                   0.0F);
  for (int column = 0; column < column_count; ++column) {
    column_widths[static_cast<std::size_t>(column)] =
        ImGui::CalcTextSize(
            snapshot.columns[static_cast<std::size_t>(column)].c_str())
            .x;
  }
  for (const auto &row : frame.rows) {
    const auto cell_count = std::min<std::size_t>(
        row.size(), static_cast<std::size_t>(column_count));
    for (std::size_t column = 0; column < cell_count; ++column) {
      column_widths[column] =
          std::max(column_widths[column],
                   ImGui::CalcTextSize(row[column].text.c_str()).x);
    }
  }
  float content_width = static_cast<float>(column_count + 1) +
                        style.ScrollbarSize; // borders + ScrollY bar
  for (auto &width : column_widths) {
    width += style.CellPadding.x * 2.0F;
    content_width += width;
  }

  constexpr auto slider_width = 20.0F;
  const auto text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  const auto table_width =
      std::max(120.0F, ImGui::GetContentRegionAvail().x - slider_width -
                           style.ItemSpacing.x);
  // Fill the remaining window height (minus the title just drawn); the table
  // scrolls vertically only when the rows genuinely overflow that height.
  const auto table_height =
      std::max(120.0F, height_budget - (ImGui::GetCursorPosY() - content_top));
  // Stretch the columns to fill the width when they fit -- no empty column and
  // no horizontal scrollbar. Fall back to fixed widths + horizontal scroll only
  // when the content is genuinely wider than the region (e.g. hundreds of
  // units).
  const bool needs_hscroll = content_width > table_width;

  const auto has_sortable_columns = [&] {
    for (std::size_t column = 0; column < snapshot.columns.size(); ++column) {
      if (column_is_sortable(frame, column)) {
        return true;
      }
    }
    return false;
  }();

  auto flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
               ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
  if (has_sortable_columns) {
    flags |= ImGuiTableFlags_Sortable;
  }
  flags |= needs_hscroll
               ? (ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit)
               : ImGuiTableFlags_SizingStretchProp;
  if (ImGui::BeginTable("##scoretable", column_count, flags,
                        ImVec2(table_width, table_height))) {
    // Freeze the header row (and the corner column too when horizontally
    // scrolling).
    ImGui::TableSetupScrollFreeze(needs_hscroll ? 1 : 0, 1);
    const auto column_flag = needs_hscroll ? ImGuiTableColumnFlags_WidthFixed
                                           : ImGuiTableColumnFlags_WidthStretch;
    for (int column = 0; column < column_count; ++column) {
      ImGuiTableColumnFlags setup_flags = column_flag;
      const auto column_index = static_cast<std::size_t>(column);
      if (has_sortable_columns && !column_is_sortable(frame, column_index)) {
        setup_flags |= ImGuiTableColumnFlags_NoSort;
      }
      if (snapshot.sort.has_value() &&
          snapshot.sort->column_index == column_index) {
        setup_flags |= ImGuiTableColumnFlags_DefaultSort;
        setup_flags |= snapshot.sort->direction == TableSortDirection::Ascending
                           ? ImGuiTableColumnFlags_PreferSortAscending
                           : ImGuiTableColumnFlags_PreferSortDescending;
      }
      ImGui::TableSetupColumn(snapshot.columns[column_index].c_str(),
                              setup_flags, column_widths[column_index],
                              static_cast<ImGuiID>(column_index + 1));
    }

    if (has_sortable_columns) {
      // Real ImGui headers own the ascending/descending interaction and
      // arrow. The generic view translates that state into a stable row
      // order; it never interprets what a column means.
      ImGui::TableHeadersRow();
      if (auto *sort_specs = ImGui::TableGetSortSpecs();
          sort_specs != nullptr && sort_specs->SpecsCount > 0) {
        const auto &first = sort_specs->Specs[0];
        snapshot.sort = TableSortSpec{
            .column_index = static_cast<std::size_t>(first.ColumnIndex),
            .direction = first.SortDirection == ImGuiSortDirection_Descending
                             ? TableSortDirection::Descending
                             : TableSortDirection::Ascending,
        };
        sort_specs->SpecsDirty = false;
      }
    } else {
      // Preserve the centered, non-interactive ScoreTable header.
      ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
      ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                             ImGui::GetColorU32(ImGuiCol_TableHeaderBg));
      for (int column = 0; column < column_count; ++column) {
        ImGui::TableSetColumnIndex(column);
        draw_centered_text(
            snapshot.columns[static_cast<std::size_t>(column)].c_str(),
            text_color);
      }
    }

    std::vector<std::size_t> row_order(frame.rows.size());
    std::iota(row_order.begin(), row_order.end(), std::size_t{0});
    if (snapshot.sort.has_value() &&
        snapshot.sort->column_index < snapshot.columns.size() &&
        column_is_sortable(frame, snapshot.sort->column_index)) {
      row_order = table_row_order(frame, *snapshot.sort);
    }
    for (const auto row_index : row_order) {
      const auto &row = frame.rows[row_index];
      ImGui::TableNextRow();
      for (int column = 0; column < column_count; ++column) {
        ImGui::TableSetColumnIndex(column);
        if (static_cast<std::size_t>(column) >= row.size()) {
          continue;
        }
        const auto &cell = row[static_cast<std::size_t>(column)];
        if (cell.tint) {
          ImGui::TableSetBgColor(
              ImGuiTableBgTarget_CellBg,
              ImGui::GetColorU32(im_color(*cell.tint, 0.45F)));
        }
        draw_centered_text(cell.text.c_str(),
                           cell.emphasize ? ImVec4(1.0F, 1.0F, 1.0F, 1.0F)
                                          : text_color);
        if (!cell.hover.empty() && ImGui::IsItemHovered()) {
          std::string tooltip;
          for (const auto &[key, value] : cell.hover) {
            if (!tooltip.empty()) {
              tooltip += "\n";
            }
            tooltip += key + ": " + value;
          }
          ImGui::SetTooltip("%s", tooltip.c_str());
        }
      }
    }
    ImGui::EndTable();
  }

  // Always-visible vertical history scrubber, right of the table. It grows
  // down: frame 1 (oldest) at the top, the latest at the bottom; parking at the
  // bottom keeps following. Positions are 1-based.
  ImGui::SameLine();
  const auto position = frame_index + 1;
  auto slider_value = frame_count - position + 1; // invert so 1 is at the top
  ImGui::BeginDisabled(frame_count < 2);
  if (ImGui::VSliderInt("##history", ImVec2(slider_width, table_height),
                        &slider_value, 1, std::max(frame_count, 1), "")) {
    const auto new_position = frame_count - slider_value + 1;
    snapshot.cursor = new_position >= frame_count ? -1 : new_position - 1;
  }
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("history %lld (%d / %d)\n%lld - %lld",
                      static_cast<long long>(frame.n), position, frame_count,
                      static_cast<long long>(snapshot.frames.front().n),
                      static_cast<long long>(snapshot.frames.back().n));
  }

  ImGui::PopID();
  return false;
}

void draw_table_group_window(std::string_view group,
                             std::vector<TableSnapshot *> &snapshots,
                             int group_index,
                             std::vector<std::uint64_t> &clear_ids) {
  std::string title;
  for (const auto *snapshot : snapshots) {
    if (!snapshot->title.empty()) {
      title = snapshot->title;
      break;
    }
  }
  if (title.empty()) {
    title = group.empty() || group == "default" ? "Table" : std::string(group);
  }

  const auto window_id = title + "##scoretable_group_" + std::string(group);
  ImGui::SetNextWindowPos(
      ImVec2(72.0F + 36.0F * static_cast<float>(group_index),
             72.0F + 36.0F * static_cast<float>(group_index)),
      ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(720.0F, 520.0F), ImGuiCond_FirstUseEver);
  if (ImGui::Begin(window_id.c_str())) {
    const auto group_text = "group: " + std::string(group);
    ImGui::TextUnformatted(group_text.c_str());
    // Split the remaining window height evenly so each stacked table fills its
    // share (one table fills the whole window) instead of a fixed-size box.
    const auto count = static_cast<float>(snapshots.size());
    const auto spacing = ImGui::GetStyle().ItemSpacing.y;
    const auto budget =
        (ImGui::GetContentRegionAvail().y - (count - 1.0F) * spacing) / count;
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
      if (index != 0) {
        ImGui::Spacing();
      }
      if (draw_table_snapshot(*snapshots[index], budget)) {
        clear_ids.push_back(snapshots[index]->id);
      }
    }
  }
  ImGui::End();
}

} // namespace

std::vector<std::size_t> table_row_order(const TableFrame &frame,
                                         const TableSortSpec &sort) {
  std::vector<std::size_t> order(frame.rows.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::size_t left_index, std::size_t right_index) {
        const auto *left =
            sort_value_at(frame.rows[left_index], sort.column_index);
        const auto *right =
            sort_value_at(frame.rows[right_index], sort.column_index);
        const auto left_missing = left == nullptr || sort_value_missing(*left);
        const auto right_missing =
            right == nullptr || sort_value_missing(*right);
        if (left_missing || right_missing) {
          if (left_missing == right_missing) {
            return false;
          }
          return !left_missing; // unavailable values stay last in both
                                // directions
        }
        const auto comparison = compare_sort_values(*left, *right);
        return sort.direction == TableSortDirection::Ascending ? comparison < 0
                                                               : comparison > 0;
      });
  return order;
}

void TableView::push(std::string element_name, const TableUpdate &update) {
  const auto lock = std::scoped_lock(mutex_);
  auto group = update.group.empty() ? std::string("default") : update.group;

  if (update.update_mode == TableUpdateMode::AppendRows) {
    static_cast<void>(column_indexes(update.columns, "AppendRows"));
  }

  TableSnapshot *snapshot = nullptr;
  for (auto &existing : snapshots_) {
    if (existing.element_name == element_name) {
      snapshot = &existing;
      break;
    }
  }
  if (snapshot == nullptr) {
    TableSnapshot fresh;
    fresh.id = next_id_++;
    fresh.element_name = std::move(element_name);
    snapshot = &snapshots_.emplace_back(std::move(fresh));
  }
  snapshot->group = std::move(group);
  snapshot->title = update.title;
  const auto previous_columns = snapshot->columns;
  switch (update.update_mode) {
  case TableUpdateMode::AppendFrame:
    if (snapshot->columns != update.columns && !snapshot->frames.empty()) {
      // Headers are snapshot-wide; do not reinterpret retained history using a
      // different positional schema.
      snapshot->frames.clear();
    }
    snapshot->columns = update.columns;
    snapshot->frames.push_back(update.frame);
    break;
  case TableUpdateMode::ReplaceFrame:
    if (snapshot->columns != update.columns && snapshot->frames.size() > 1) {
      // Column headers are snapshot-wide. Starting a new schema cannot
      // leave older history frames interpreted under unrelated headers.
      snapshot->frames.clear();
    }
    snapshot->columns = update.columns;
    if (snapshot->frames.empty()) {
      snapshot->frames.push_back(update.frame);
    } else {
      snapshot->frames.back() = update.frame;
    }
    break;
  case TableUpdateMode::AppendRows:
    if (snapshot->frames.empty()) {
      snapshot->columns = update.columns;
      snapshot->frames.push_back(update.frame);
      break;
    }
    static_cast<void>(column_indexes(snapshot->columns, "AppendRows"));
    {
      auto union_columns = snapshot->columns;
      auto union_indexes = column_indexes(union_columns, "AppendRows");
      for (const auto &column : update.columns) {
        if (!union_indexes.contains(column)) {
          union_indexes.emplace(column, union_columns.size());
          union_columns.push_back(column);
        }
      }
      if (union_columns != snapshot->columns) {
        for (auto &frame : snapshot->frames) {
          frame = remap_frame_columns(frame, snapshot->columns, union_columns);
        }
      }
      auto incoming =
          remap_frame_columns(update.frame, update.columns, union_columns);
      auto &current = snapshot->frames.back();
      current.n = incoming.n;
      current.caption = std::move(incoming.caption);
      current.rows.insert(current.rows.end(),
                          std::make_move_iterator(incoming.rows.begin()),
                          std::make_move_iterator(incoming.rows.end()));
      snapshot->columns = std::move(union_columns);
    }
    break;
  default:
    throw std::invalid_argument("TableView update mode is invalid");
  }

  if (update.update_mode != TableUpdateMode::AppendRows &&
      previous_columns != snapshot->columns) {
    snapshot->sort.reset();
  }
  if (snapshot->sort.has_value() &&
      snapshot->sort->column_index >= snapshot->columns.size()) {
    snapshot->sort.reset();
  }
  // Trim history: 0 keeps everything, otherwise cap to max_history (1 =
  // replace).
  if (update.max_history != 0) {
    while (snapshot->frames.size() > update.max_history) {
      snapshot->frames.pop_front();
    }
  }
}

bool TableView::erase(std::string_view element_name) {
  const auto lock = std::scoped_lock(mutex_);
  const auto previous_size = snapshots_.size();
  std::erase_if(snapshots_, [element_name](const auto &snapshot) {
    return snapshot.element_name == element_name;
  });
  return snapshots_.size() != previous_size;
}

bool TableView::update_presentation(std::string_view element_name,
                                    std::string group, std::string title) {
  const auto lock = std::scoped_lock(mutex_);
  const auto found =
      std::ranges::find_if(snapshots_, [element_name](const auto &snapshot) {
        return snapshot.element_name == element_name;
      });
  if (found == snapshots_.end()) {
    return false;
  }
  found->group = group.empty() ? std::string("default") : std::move(group);
  found->title = std::move(title);
  return true;
}

const std::vector<TableSnapshot> &TableView::snapshots() const {
  const auto lock = std::scoped_lock(mutex_);
  return snapshots_;
}

std::vector<TableSnapshot> TableView::snapshots_copy() const {
  const auto lock = std::scoped_lock(mutex_);
  return snapshots_;
}

void TableView::draw(const PlotDrawContext & /*context*/) {
  const auto lock = std::scoped_lock(mutex_);
  std::map<std::string, std::vector<TableSnapshot *>> groups;
  for (auto &snapshot : snapshots_) {
    groups[snapshot.group].push_back(&snapshot);
  }
  auto group_index = 0;
  std::vector<std::uint64_t> clear_ids;
  for (auto &[group, snapshots] : groups) {
    draw_table_group_window(group, snapshots, group_index, clear_ids);
    ++group_index;
  }
  if (!clear_ids.empty()) {
    std::erase_if(snapshots_, [&clear_ids](const auto &snapshot) {
      return std::ranges::find(clear_ids, snapshot.id) != clear_ids.end();
    });
  }
}

void TableView::clear() {
  const auto lock = std::scoped_lock(mutex_);
  snapshots_.clear();
  next_id_ = 1;
}

bool TableView::empty() const {
  const auto lock = std::scoped_lock(mutex_);
  return snapshots_.empty();
}

} // namespace leakflow::plot
