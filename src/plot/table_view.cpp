#include "leakflow/plot/table_view.hpp"

#include "plot_render_util.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace leakflow::plot {

double table_heatmap_row_axis_position(std::size_t row_index,
                                       std::size_t row_count) {
  if (row_count == 0 || row_index >= row_count) {
    throw std::out_of_range("TableView heatmap row index is out of range");
  }
  return static_cast<double>(row_count - row_index) - 0.5;
}

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

void validate_row_selector(const std::vector<std::string> &columns,
                           const TableFrame &frame,
                           const std::optional<TableRowSelector> &selector) {
  if (!selector) {
    return;
  }
  if (selector->key.empty()) {
    throw std::invalid_argument(
        "TableView row selector requires a non-empty shared key");
  }
  if (selector->column.empty() ||
      std::ranges::count(columns, selector->column) != 1) {
    throw std::invalid_argument(
        "TableView row selector requires exactly one matching column");
  }
  for (std::size_t left = 0; left < selector->values.size(); ++left) {
    for (std::size_t right = left + 1; right < selector->values.size();
         ++right) {
      if (selector->values[left].value == selector->values[right].value) {
        throw std::invalid_argument(
            "TableView row selector requires unique typed values");
      }
    }
  }
  const auto column = static_cast<std::size_t>(std::distance(
      columns.begin(), std::ranges::find(columns, selector->column)));
  for (const auto &row : frame.rows) {
    if (column >= row.size() || !row[column].sort_value ||
        std::ranges::none_of(selector->values, [&](const auto &choice) {
          return choice.value == *row[column].sort_value;
        })) {
      throw std::invalid_argument(
          "TableView row selector rows require a declared typed value");
    }
  }
}

void validate_heatmap_frame(const TableFrame &frame,
                            const std::optional<TableRowSelector> &selector) {
  if (!frame.heatmap) {
    return;
  }
  const auto &pages = frame.heatmap->pages;
  if (pages.empty()) {
    throw std::invalid_argument(
        "TableView heatmap frames require at least one page");
  }
  if (!selector && pages.size() != 1) {
    throw std::invalid_argument(
        "TableView heatmap frames without a selector require one page");
  }
  if (selector && pages.size() != selector->values.size()) {
    throw std::invalid_argument(
        "TableView heatmap pages must match the selector choices");
  }

  for (std::size_t index = 0; index < pages.size(); ++index) {
    const auto &page = pages[index];
    if (std::ranges::any_of(
            pages | std::views::take(index), [&](const auto &previous) {
              return previous.selector_value == page.selector_value;
            })) {
      throw std::invalid_argument(
          "TableView heatmap pages require unique selector values");
    }
    if (selector &&
        std::ranges::none_of(selector->values, [&](const auto &choice) {
          return choice.value == page.selector_value;
        })) {
      throw std::invalid_argument(
          "TableView heatmap page uses an undeclared selector value");
    }

    if (!page.unavailable_reason.empty()) {
      if (page.rows != 0 || page.cols != 0 || !page.data.empty()) {
        throw std::invalid_argument(
            "TableView unavailable heatmap pages cannot carry matrix data");
      }
      continue;
    }
    if (page.rows <= 0 || page.cols <= 0) {
      throw std::invalid_argument(
          "TableView heatmap page dimensions must be positive");
    }
    const auto rows = static_cast<std::size_t>(page.rows);
    const auto cols = static_cast<std::size_t>(page.cols);
    if (rows > std::numeric_limits<std::size_t>::max() / cols ||
        page.data.size() != rows * cols) {
      throw std::invalid_argument(
          "TableView heatmap page data does not match its shape");
    }
    if ((!page.row_labels.empty() && page.row_labels.size() != rows) ||
        (!page.col_labels.empty() && page.col_labels.size() != cols)) {
      throw std::invalid_argument(
          "TableView heatmap labels do not match its shape");
    }
    if (!std::isfinite(page.vmin) || !std::isfinite(page.vmax) ||
        page.vmax <= page.vmin) {
      throw std::invalid_argument(
          "TableView heatmap scale must be finite and increasing");
    }
  }
}

[[nodiscard]] std::size_t row_selector_column(const TableSnapshot &snapshot) {
  const auto found =
      std::ranges::find(snapshot.columns, snapshot.row_selector->column);
  return static_cast<std::size_t>(
      std::distance(snapshot.columns.begin(), found));
}

[[nodiscard]] bool
same_row_selector_identity(const std::optional<TableRowSelector> &left,
                           const std::optional<TableRowSelector> &right) {
  if (left.has_value() != right.has_value()) {
    return false;
  }
  return !left || (left->key == right->key && left->column == right->column);
}

void merge_row_selector_values(TableRowSelector &target,
                               const TableRowSelector &source) {
  target.label = source.label;
  for (const auto &choice : source.values) {
    const auto existing = std::ranges::find(target.values, choice.value,
                                            &TableRowSelectorValue::value);
    if (existing == target.values.end()) {
      target.values.push_back(choice);
    } else {
      // The typed identity and established slider order are stable across
      // history, while the latest producer-supplied display label wins.
      existing->label = choice.label;
    }
  }
}

void prune_selected_row_values(
    const std::vector<TableSnapshot> &snapshots,
    std::map<std::string, TableCell::SortValue> &selected_values) {
  std::erase_if(selected_values, [&snapshots](const auto &selected) {
    return std::ranges::none_of(snapshots, [&selected](const auto &snapshot) {
      return snapshot.row_selector &&
             snapshot.row_selector->key == selected.first &&
             std::ranges::any_of(snapshot.row_selector->values,
                                 [&selected](const auto &choice) {
                                   return choice.value == selected.second;
                                 });
    });
  });
}

using RowSelectorValues =
    std::map<std::string, std::vector<TableRowSelectorValue>>;

[[nodiscard]] RowSelectorValues
shared_row_selector_values(const std::vector<TableSnapshot> &snapshots) {
  RowSelectorValues result;
  for (const auto &snapshot : snapshots) {
    if (!snapshot.row_selector) {
      continue;
    }
    auto &values = result[snapshot.row_selector->key];
    for (const auto &choice : snapshot.row_selector->values) {
      if (std::ranges::none_of(values, [&](const auto &existing) {
            return existing.value == choice.value;
          })) {
        values.push_back(choice);
      }
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

void draw_row_selector_slider(int &selected_index,
                              const std::vector<TableRowSelectorValue> &choices,
                              std::string_view selector_label) {
  if (choices.size() <= 1) {
    return;
  }

  ImGui::SetNextItemWidth(-1.0F);
  ImGui::SliderInt("##row_selector", &selected_index, 0,
                   static_cast<int>(choices.size()) - 1, "");
  selected_index =
      std::clamp(selected_index, 0, static_cast<int>(choices.size()) - 1);

  // The slider position and logical value are different concepts: index 0 may
  // select Unit 3. Draw both from the post-interaction selection so the label
  // and filtered rows change together in the same frame.
  const auto &selected_choice =
      choices[static_cast<std::size_t>(selected_index)];
  const auto caption = "[" + std::to_string(selected_index) +
                       "]: " + std::string(selector_label) + " " +
                       selected_choice.label;
  const auto item_min = ImGui::GetItemRectMin();
  const auto item_max = ImGui::GetItemRectMax();
  const auto text_size = ImGui::CalcTextSize(caption.c_str());
  const auto text_position = ImVec2(
      item_min.x +
          std::max(0.0F, (item_max.x - item_min.x - text_size.x) * 0.5F),
      item_min.y +
          std::max(0.0F, (item_max.y - item_min.y - text_size.y) * 0.5F));
  ImGui::GetWindowDrawList()->AddText(
      text_position, ImGui::GetColorU32(ImGuiCol_Text), caption.c_str());
}

struct HeatmapAxisTicks {
  std::vector<double> positions;
  std::vector<const char *> labels;
};

[[nodiscard]] HeatmapAxisTicks
heatmap_axis_ticks(const std::vector<std::string> &labels,
                   bool first_label_at_axis_max) {
  HeatmapAxisTicks result;
  if (labels.empty()) {
    return result;
  }
  const auto position = [&](std::size_t index) {
    return first_label_at_axis_max
               ? table_heatmap_row_axis_position(index, labels.size())
               : static_cast<double>(index) + 0.5;
  };
  constexpr std::size_t max_visible_ticks = 18;
  const auto step = std::max<std::size_t>(
      1, (labels.size() + max_visible_ticks - 1) / max_visible_ticks);
  for (std::size_t index = 0; index < labels.size(); index += step) {
    result.positions.push_back(position(index));
    result.labels.push_back(labels[index].c_str());
  }
  if ((labels.size() - 1) % step != 0) {
    result.positions.push_back(position(labels.size() - 1));
    result.labels.push_back(labels.back().c_str());
  }
  return result;
}

void draw_heatmap_frame(
    const TableHeatmapFrame &heatmap,
    const std::optional<TableCell::SortValue> &selected_value, float width,
    float height) {
  auto selected = selected_value ? heatmap.pages.end() : heatmap.pages.begin();
  if (selected_value) {
    selected = std::ranges::find(heatmap.pages, *selected_value,
                                 &TableHeatmapPage::selector_value);
  }
  if (selected == heatmap.pages.end()) {
    if (ImGui::BeginChild("##heatmap_missing_page", ImVec2(width, height),
                          true)) {
      ImGui::TextWrapped(
          "Unavailable: the selected value has no matrix in this history "
          "frame");
    }
    ImGui::EndChild();
    return;
  }

  ImGui::BeginGroup();
  const auto content_top = ImGui::GetCursorPosY();
  if (!selected->caption.empty()) {
    ImGui::TextUnformatted(selected->caption.c_str());
  }
  if (!selected->unavailable_reason.empty()) {
    const auto unavailable_height =
        std::max(120.0F, height - (ImGui::GetCursorPosY() - content_top));
    if (ImGui::BeginChild("##heatmap_unavailable",
                          ImVec2(width, unavailable_height), true)) {
      ImGui::TextWrapped("Unavailable: %s",
                         selected->unavailable_reason.c_str());
    }
    ImGui::EndChild();
    ImGui::EndGroup();
    return;
  }

  // PlotHeatmap already maps row zero to the maximum Y bound. Keep the normal
  // Y axis and place row-label ticks in the same descending coordinate order;
  // adding ImPlotAxisFlags_Invert here would flip cells away from their labels.
  const auto row_ticks = heatmap_axis_ticks(selected->row_labels,
                                            /*first_label_at_axis_max=*/true);
  const auto col_ticks = heatmap_axis_ticks(selected->col_labels,
                                            /*first_label_at_axis_max=*/false);
  constexpr auto scale_width = 56.0F;
  const auto plot_width =
      std::max(120.0F, width - scale_width - ImGui::GetStyle().ItemSpacing.x);
  const auto plot_height =
      std::max(120.0F, height - (ImGui::GetCursorPosY() - content_top));
  ImPlot::PushColormap(ImPlotColormap_Viridis);
  if (ImPlot::BeginPlot("##heatmap", ImVec2(plot_width, plot_height),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
    const auto axis_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
    ImPlot::SetupAxes(selected->col_axis_label.c_str(),
                      selected->row_axis_label.c_str(), axis_flags, axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, static_cast<double>(selected->cols),
                            ImGuiCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, static_cast<double>(selected->rows),
                            ImGuiCond_Always);
    if (!col_ticks.positions.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_X1, col_ticks.positions.data(),
                             static_cast<int>(col_ticks.positions.size()),
                             col_ticks.labels.data());
    }
    if (!row_ticks.positions.empty()) {
      ImPlot::SetupAxisTicks(ImAxis_Y1, row_ticks.positions.data(),
                             static_cast<int>(row_ticks.positions.size()),
                             row_ticks.labels.data());
    }
    ImPlot::PlotHeatmap("##cells", selected->data.data(),
                        static_cast<int>(selected->rows),
                        static_cast<int>(selected->cols), selected->vmin,
                        selected->vmax, nullptr, ImPlotPoint(0.0, 0.0),
                        ImPlotPoint(static_cast<double>(selected->cols),
                                    static_cast<double>(selected->rows)));
    ImPlot::EndPlot();
  }
  ImGui::SameLine();
  ImPlot::ColormapScale("##scale", selected->vmin, selected->vmax,
                        ImVec2(scale_width, plot_height));
  ImPlot::PopColormap();
  ImGui::EndGroup();
}

void draw_history_slider(TableSnapshot &snapshot, const TableFrame &frame,
                         int frame_index, float height, float width) {
  const auto frame_count = static_cast<int>(snapshot.frames.size());
  const auto position = frame_index + 1;
  auto slider_value = frame_count - position + 1;
  ImGui::BeginDisabled(frame_count < 2);
  if (ImGui::VSliderInt("##history", ImVec2(width, height), &slider_value, 1,
                        std::max(frame_count, 1), "")) {
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
}

[[nodiscard]] bool draw_table_snapshot(
    TableSnapshot &snapshot, const RowSelectorValues &shared_selector_values,
    std::map<std::string, TableCell::SortValue> &selected_row_values,
    float height_budget, bool show_snapshot_title = true) {
  if (snapshot.frames.empty()) {
    return false;
  }
  // A selector-less heatmap intentionally needs no synthetic table column.
  // Resolve the active frame before rejecting an empty tabular schema so that
  // its sole page reaches the heatmap renderer.
  const auto frame_count = static_cast<int>(snapshot.frames.size());
  const auto frame_index =
      (snapshot.cursor < 0 || snapshot.cursor >= frame_count) ? frame_count - 1
                                                              : snapshot.cursor;
  const auto &frame = snapshot.frames[static_cast<std::size_t>(frame_index)];
  if (snapshot.columns.empty() && !frame.heatmap) {
    return false;
  }
  const auto content_top = ImGui::GetCursorPosY();
  ImGui::PushID(static_cast<int>(snapshot.id));
  if (show_snapshot_title && !snapshot.title.empty()) {
    ImGui::SeparatorText(snapshot.title.c_str());
  }
  if (ImGui::SmallButton("Clear")) {
    ImGui::PopID();
    return true;
  }

  // History cursor: -1 follows the latest frame; otherwise the pinned frame
  // index.
  std::optional<std::size_t> selector_column;
  std::optional<TableCell::SortValue> selected_row_value;
  if (snapshot.row_selector) {
    selector_column = row_selector_column(snapshot);
    const auto choices =
        shared_selector_values.find(snapshot.row_selector->key);
    if (choices != shared_selector_values.end() && !choices->second.empty()) {
      auto selected = selected_row_values.find(snapshot.row_selector->key);
      auto selected_position =
          selected == selected_row_values.end()
              ? choices->second.end()
              : std::ranges::find(choices->second, selected->second,
                                  &TableRowSelectorValue::value);
      if (selected_position == choices->second.end()) {
        selected = selected_row_values
                       .insert_or_assign(snapshot.row_selector->key,
                                         choices->second.front().value)
                       .first;
        selected_position = choices->second.begin();
      }
      auto selected_index = static_cast<int>(
          std::distance(choices->second.begin(), selected_position));
      const auto &selector_label = snapshot.row_selector->label.empty()
                                       ? snapshot.row_selector->column
                                       : snapshot.row_selector->label;
      draw_row_selector_slider(selected_index, choices->second, selector_label);
      const auto &selected_choice =
          choices->second[static_cast<std::size_t>(selected_index)];
      selected->second = selected_choice.value;
      selected_row_value = selected_choice.value;
    } else {
      selector_column.reset();
    }
  }

  constexpr auto slider_width = 20.0F;
  const auto &style = ImGui::GetStyle();
  if (frame.heatmap) {
    const auto content_width =
        std::max(120.0F, ImGui::GetContentRegionAvail().x - slider_width -
                             style.ItemSpacing.x);
    const auto content_height = std::max(
        120.0F, height_budget - (ImGui::GetCursorPosY() - content_top));
    draw_heatmap_frame(*frame.heatmap, selected_row_value, content_width,
                       content_height);
    ImGui::SameLine();
    draw_history_slider(snapshot, frame, frame_index, content_height,
                        slider_width);
    ImGui::PopID();
    return false;
  }

  const auto column_count = static_cast<int>(snapshot.columns.size());

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
      if (selector_column &&
          !table_row_matches_selector(frame, row_index, *selector_column,
                                      *selected_row_value)) {
        continue;
      }
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

  // Always-visible vertical history scrubber, right of the content. It grows
  // down: frame 1 (oldest) at the top, the latest at the bottom; parking at the
  // bottom keeps following. Positions are 1-based.
  ImGui::SameLine();
  draw_history_slider(snapshot, frame, frame_index, table_height, slider_width);

  ImGui::PopID();
  return false;
}

void draw_table_group_window(
    std::string_view group, std::vector<TableSnapshot *> &snapshots,
    int group_index, const RowSelectorValues &shared_selector_values,
    std::map<std::string, TableCell::SortValue> &selected_row_values,
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
      if (draw_table_snapshot(*snapshots[index], shared_selector_values,
                              selected_row_values, budget)) {
        clear_ids.push_back(snapshots[index]->id);
      }
    }
  }
  ImGui::End();
}

[[nodiscard]] std::string tab_label_for(const TableSnapshot &snapshot) {
  if (!snapshot.tab_label.empty()) {
    return snapshot.tab_label;
  }
  if (!snapshot.title.empty()) {
    return snapshot.title;
  }
  if (!snapshot.element_name.empty()) {
    return snapshot.element_name;
  }
  return "Table";
}

void draw_tabbed_table_group_window(
    std::string_view group, std::vector<TableSnapshot *> &snapshots,
    int group_index, const RowSelectorValues &shared_selector_values,
    std::map<std::string, TableCell::SortValue> &selected_row_values,
    std::vector<std::uint64_t> &clear_ids) {
  std::stable_sort(snapshots.begin(), snapshots.end(),
                   [](const auto *left, const auto *right) {
                     return left->tab_order < right->tab_order;
                   });
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

  // Tabs and stacked snapshots with the same logical group deliberately use
  // different hidden window IDs, so mixed layouts never share ImGui state.
  const auto window_id = title + "##table_tabs_group_" + std::string(group);
  ImGui::SetNextWindowPos(
      ImVec2(72.0F + 36.0F * static_cast<float>(group_index),
             72.0F + 36.0F * static_cast<float>(group_index)),
      ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(720.0F, 520.0F), ImGuiCond_FirstUseEver);
  if (ImGui::Begin(window_id.c_str())) {
    const auto group_text = "group: " + std::string(group);
    ImGui::TextUnformatted(group_text.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear all")) {
      for (const auto *snapshot : snapshots) {
        clear_ids.push_back(snapshot->id);
      }
    }

    if (ImGui::BeginTabBar("##table_group_tabs")) {
      std::map<std::string, std::size_t> label_counts;
      for (const auto *snapshot : snapshots) {
        ++label_counts[tab_label_for(*snapshot)];
      }
      for (auto *snapshot : snapshots) {
        auto visible_label = tab_label_for(*snapshot);
        // A comparison group may contain more than one producer using the
        // same domain label (for example, two "Overview" tabs). Keep the
        // common single-producer case terse and disambiguate duplicates with
        // the generic producer identity.
        if (label_counts.at(visible_label) > 1) {
          visible_label += " (" +
                           (snapshot->element_name.empty()
                                ? "table " + std::to_string(snapshot->id)
                                : snapshot->element_name) +
                           ")";
        }
        const auto tab_id =
            visible_label + "###table_tab_" + std::to_string(snapshot->id);
        if (ImGui::BeginTabItem(tab_id.c_str())) {
          const auto height_budget = ImGui::GetContentRegionAvail().y;
          if (draw_table_snapshot(*snapshot, shared_selector_values,
                                  selected_row_values, height_budget, false)) {
            clear_ids.push_back(snapshot->id);
          }
          ImGui::EndTabItem();
        }
      }
      ImGui::EndTabBar();
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

bool table_row_matches_selector(const TableFrame &frame, std::size_t row_index,
                                std::size_t column_index,
                                const TableCell::SortValue &selected_value) {
  return row_index < frame.rows.size() &&
         column_index < frame.rows[row_index].size() &&
         frame.rows[row_index][column_index].sort_value.has_value() &&
         *frame.rows[row_index][column_index].sort_value == selected_value;
}

void TableView::push(std::string element_name, const TableUpdate &update) {
  const auto lock = std::scoped_lock(mutex_);
  auto group = update.group.empty() ? std::string("default") : update.group;

  validate_row_selector(update.columns, update.frame, update.row_selector);
  validate_heatmap_frame(update.frame, update.row_selector);

  if (update.update_mode == TableUpdateMode::AppendRows) {
    if (update.frame.heatmap) {
      throw std::invalid_argument(
          "TableView AppendRows does not accept heatmap frames");
    }
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
  const auto previous_columns = snapshot->columns;
  const auto previous_selector = snapshot->row_selector;
  const auto selector_schema_changed =
      !same_row_selector_identity(previous_selector, update.row_selector);
  const auto content_schema_changed =
      !snapshot->frames.empty() &&
      snapshot->frames.back().heatmap.has_value() !=
          update.frame.heatmap.has_value();
  if (update.update_mode == TableUpdateMode::AppendRows &&
      !snapshot->frames.empty() &&
      (selector_schema_changed || snapshot->frames.back().heatmap)) {
    throw std::invalid_argument(
        "TableView AppendRows cannot change selector/content schema");
  }
  auto next_selector = update.row_selector;
  if (!selector_schema_changed && previous_selector && next_selector &&
      update.update_mode != TableUpdateMode::ReplaceFrame &&
      (update.update_mode == TableUpdateMode::AppendRows ||
       previous_columns == update.columns)) {
    next_selector = previous_selector;
    merge_row_selector_values(*next_selector, *update.row_selector);
  }
  snapshot->group = std::move(group);
  snapshot->group_layout = update.group_layout;
  snapshot->tab_label = update.tab_label;
  snapshot->tab_order = update.tab_order;
  snapshot->row_selector = std::move(next_selector);
  snapshot->title = update.title;
  switch (update.update_mode) {
  case TableUpdateMode::AppendFrame:
    if ((snapshot->columns != update.columns || selector_schema_changed ||
         content_schema_changed) &&
        !snapshot->frames.empty()) {
      // Headers and selector schema are snapshot-wide; do not reinterpret
      // retained history using unrelated presentation metadata.
      snapshot->frames.clear();
      snapshot->cursor = -1;
    }
    snapshot->columns = update.columns;
    snapshot->frames.push_back(update.frame);
    break;
  case TableUpdateMode::ReplaceFrame:
    if ((snapshot->columns != update.columns || selector_schema_changed ||
         content_schema_changed) &&
        snapshot->frames.size() > 1) {
      // Column headers and selector schema are snapshot-wide. Starting a new
      // schema cannot leave older history interpreted under unrelated metadata.
      snapshot->frames.clear();
    }
    snapshot->cursor = -1;
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
    std::size_t removed = 0;
    while (snapshot->frames.size() > update.max_history) {
      snapshot->frames.pop_front();
      ++removed;
    }
    if (snapshot->cursor >= 0 && removed != 0) {
      const auto old_cursor = static_cast<std::size_t>(snapshot->cursor);
      snapshot->cursor =
          old_cursor < removed ? 0 : static_cast<int>(old_cursor - removed);
    }
    if (snapshot->cursor >= static_cast<int>(snapshot->frames.size())) {
      snapshot->cursor = -1;
    }
  }
  prune_selected_row_values(snapshots_, selected_row_values_);
}

bool TableView::erase(std::string_view element_name) {
  const auto lock = std::scoped_lock(mutex_);
  const auto previous_size = snapshots_.size();
  std::erase_if(snapshots_, [element_name](const auto &snapshot) {
    return snapshot.element_name == element_name;
  });
  prune_selected_row_values(snapshots_, selected_row_values_);
  return snapshots_.size() != previous_size;
}

bool TableView::contains(std::string_view element_name) const {
  const auto lock = std::scoped_lock(mutex_);
  return std::ranges::any_of(snapshots_, [element_name](const auto &snapshot) {
    return snapshot.element_name == element_name;
  });
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

bool TableView::select_row_value(std::string_view selector_key,
                                 TableCell::SortValue value) {
  const auto lock = std::scoped_lock(mutex_);
  const auto available =
      std::ranges::any_of(snapshots_, [&](const auto &snapshot) {
        return snapshot.row_selector &&
               snapshot.row_selector->key == selector_key &&
               std::ranges::any_of(
                   snapshot.row_selector->values,
                   [&](const auto &choice) { return choice.value == value; });
      });
  if (!available) {
    return false;
  }
  selected_row_values_.insert_or_assign(std::string(selector_key),
                                        std::move(value));
  return true;
}

std::optional<TableCell::SortValue>
TableView::selected_row_value(std::string_view selector_key) const {
  const auto lock = std::scoped_lock(mutex_);
  const auto selected = selected_row_values_.find(std::string(selector_key));
  return selected == selected_row_values_.end()
             ? std::nullopt
             : std::optional<TableCell::SortValue>(selected->second);
}

bool TableView::select_history_frame(std::string_view element_name,
                                     std::optional<std::size_t> frame_index) {
  const auto lock = std::scoped_lock(mutex_);
  const auto snapshot =
      std::ranges::find_if(snapshots_, [element_name](const auto &candidate) {
        return candidate.element_name == element_name;
      });
  if (snapshot == snapshots_.end() || snapshot->frames.empty() ||
      (frame_index && *frame_index >= snapshot->frames.size())) {
    return false;
  }
  snapshot->cursor = !frame_index || *frame_index == snapshot->frames.size() - 1
                         ? -1
                         : static_cast<int>(*frame_index);
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
  const auto selector_values = shared_row_selector_values(snapshots_);
  std::map<std::string, std::vector<TableSnapshot *>> stacked_groups;
  std::map<std::string, std::vector<TableSnapshot *>> tabbed_groups;
  for (auto &snapshot : snapshots_) {
    auto &groups = snapshot.group_layout == TableGroupLayout::Tabs
                       ? tabbed_groups
                       : stacked_groups;
    groups[snapshot.group].push_back(&snapshot);
  }
  auto group_index = 0;
  std::vector<std::uint64_t> clear_ids;
  for (auto &[group, snapshots] : stacked_groups) {
    draw_table_group_window(group, snapshots, group_index, selector_values,
                            selected_row_values_, clear_ids);
    ++group_index;
  }
  for (auto &[group, snapshots] : tabbed_groups) {
    draw_tabbed_table_group_window(group, snapshots, group_index,
                                   selector_values, selected_row_values_,
                                   clear_ids);
    ++group_index;
  }
  if (!clear_ids.empty()) {
    std::erase_if(snapshots_, [&clear_ids](const auto &snapshot) {
      return std::ranges::find(clear_ids, snapshot.id) != clear_ids.end();
    });
    prune_selected_row_values(snapshots_, selected_row_values_);
  }
}

void TableView::clear() {
  const auto lock = std::scoped_lock(mutex_);
  snapshots_.clear();
  selected_row_values_.clear();
  next_id_ = 1;
}

bool TableView::empty() const {
  const auto lock = std::scoped_lock(mutex_);
  return snapshots_.empty();
}

} // namespace leakflow::plot
