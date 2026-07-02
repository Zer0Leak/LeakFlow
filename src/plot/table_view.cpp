#include "leakflow/plot/table_view.hpp"

#include "plot_render_util.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow::plot {
namespace {

// Center a cell's/header's text within the current table column (ImGui aligns left by
// default). Leaves the drawn text as the "last item" so IsItemHovered still works.
void draw_centered_text(const char *text, const ImVec4 &color) {
    const auto text_width = ImGui::CalcTextSize(text).x;
    const auto avail = ImGui::GetContentRegionAvail().x;
    if (avail > text_width) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - text_width) * 0.5F);
    }
    ImGui::TextColored(color, "%s", text);
}

void draw_table_snapshot(TableSnapshot &snapshot, float height_budget) {
    if (snapshot.frames.empty() || snapshot.columns.empty()) {
        return;
    }
    const auto content_top = ImGui::GetCursorPosY();
    if (!snapshot.title.empty()) {
        ImGui::SeparatorText(snapshot.title.c_str());
    }

    // History cursor: -1 follows the latest frame; otherwise the pinned frame index.
    const auto frame_count = static_cast<int>(snapshot.frames.size());
    const auto frame_index =
        (snapshot.cursor < 0 || snapshot.cursor >= frame_count) ? frame_count - 1 : snapshot.cursor;
    const auto &frame = snapshot.frames[static_cast<std::size_t>(frame_index)];

    const auto column_count = static_cast<int>(snapshot.columns.size());
    const auto &style = ImGui::GetStyle();

    // Auto-size each column to the widest of its header + cells (so nothing clips and
    // no manual resize is needed), then size the table to that content -- a window
    // wider than the table then just leaves plain background, not an empty bordered
    // column. ScrollX still kicks in once the content is wider than the region.
    std::vector<float> column_widths(static_cast<std::size_t>(column_count), 0.0F);
    for (int column = 0; column < column_count; ++column) {
        column_widths[static_cast<std::size_t>(column)] =
            ImGui::CalcTextSize(snapshot.columns[static_cast<std::size_t>(column)].c_str()).x;
    }
    for (const auto &row : frame.rows) {
        const auto cell_count = std::min<std::size_t>(row.size(), static_cast<std::size_t>(column_count));
        for (std::size_t column = 0; column < cell_count; ++column) {
            column_widths[column] = std::max(column_widths[column], ImGui::CalcTextSize(row[column].text.c_str()).x);
        }
    }
    float content_width = static_cast<float>(column_count + 1) + style.ScrollbarSize; // borders + ScrollY bar
    for (auto &width : column_widths) {
        width += style.CellPadding.x * 2.0F;
        content_width += width;
    }

    ImGui::PushID(static_cast<int>(snapshot.id));

    constexpr auto slider_width = 20.0F;
    const auto text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    const auto table_width = std::max(120.0F, ImGui::GetContentRegionAvail().x - slider_width - style.ItemSpacing.x);
    // Fill the remaining window height (minus the title just drawn); the table scrolls
    // vertically only when the rows genuinely overflow that height.
    const auto table_height = std::max(120.0F, height_budget - (ImGui::GetCursorPosY() - content_top));
    // Stretch the columns to fill the width when they fit -- no empty column and no
    // horizontal scrollbar. Fall back to fixed widths + horizontal scroll only when
    // the content is genuinely wider than the region (e.g. hundreds of units).
    const bool needs_hscroll = content_width > table_width;

    auto flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
    flags |= needs_hscroll ? (ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit)
                           : ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("##scoretable", column_count, flags, ImVec2(table_width, table_height))) {
        // Freeze the header row (and the corner column too when horizontally scrolling).
        ImGui::TableSetupScrollFreeze(needs_hscroll ? 1 : 0, 1);
        const auto column_flag = needs_hscroll ? ImGuiTableColumnFlags_WidthFixed : ImGuiTableColumnFlags_WidthStretch;
        for (int column = 0; column < column_count; ++column) {
            ImGui::TableSetupColumn(snapshot.columns[static_cast<std::size_t>(column)].c_str(), column_flag,
                                    column_widths[static_cast<std::size_t>(column)]);
        }

        // Centered header row (ImGui's default header row is left-aligned).
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TableHeaderBg));
        for (int column = 0; column < column_count; ++column) {
            ImGui::TableSetColumnIndex(column);
            draw_centered_text(snapshot.columns[static_cast<std::size_t>(column)].c_str(), text_color);
        }

        for (const auto &row : frame.rows) {
            ImGui::TableNextRow();
            for (int column = 0; column < column_count; ++column) {
                ImGui::TableSetColumnIndex(column);
                if (static_cast<std::size_t>(column) >= row.size()) {
                    continue;
                }
                const auto &cell = row[static_cast<std::size_t>(column)];
                if (cell.tint) {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(im_color(*cell.tint, 0.45F)));
                }
                draw_centered_text(cell.text.c_str(), cell.emphasize ? ImVec4(1.0F, 1.0F, 1.0F, 1.0F) : text_color);
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

    // Always-visible vertical history scrubber, right of the table. It grows down:
    // frame 1 (oldest) at the top, the latest at the bottom; parking at the bottom
    // keeps following. Positions are 1-based.
    ImGui::SameLine();
    const auto position = frame_index + 1;
    auto slider_value = frame_count - position + 1; // invert so 1 is at the top
    ImGui::BeginDisabled(frame_count < 2);
    if (ImGui::VSliderInt("##history", ImVec2(slider_width, table_height), &slider_value, 1, std::max(frame_count, 1),
                          "")) {
        const auto new_position = frame_count - slider_value + 1;
        snapshot.cursor = new_position >= frame_count ? -1 : new_position - 1;
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("history %lld (%d / %d)\n%lld - %lld", static_cast<long long>(frame.n), position, frame_count,
                          static_cast<long long>(snapshot.frames.front().n),
                          static_cast<long long>(snapshot.frames.back().n));
    }

    ImGui::PopID();
}

void draw_table_group_window(std::string_view group, std::vector<TableSnapshot *> &snapshots, int group_index) {
    std::string title;
    for (const auto *snapshot : snapshots) {
        if (!snapshot->title.empty()) {
            title = snapshot->title;
            break;
        }
    }
    if (title.empty()) {
        title = group.empty() || group == "default" ? "Score table" : std::string(group);
    }

    const auto window_id = title + "##scoretable_group_" + std::string(group);
    ImGui::SetNextWindowPos(ImVec2(72.0F + 36.0F * static_cast<float>(group_index),
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
        const auto budget = (ImGui::GetContentRegionAvail().y - (count - 1.0F) * spacing) / count;
        for (std::size_t index = 0; index < snapshots.size(); ++index) {
            if (index != 0) {
                ImGui::Spacing();
            }
            draw_table_snapshot(*snapshots[index], budget);
        }
    }
    ImGui::End();
}

} // namespace

void TableView::push(std::string element_name, const TableUpdate &update) {
    const auto lock = std::scoped_lock(mutex_);
    auto group = update.group.empty() ? std::string("default") : update.group;

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
    snapshot->columns = update.columns;
    snapshot->frames.push_back(update.frame);
    // Trim history: 0 keeps everything, otherwise cap to max_history (1 = replace).
    if (update.max_history != 0) {
        while (snapshot->frames.size() > update.max_history) {
            snapshot->frames.pop_front();
        }
    }
}

const std::vector<TableSnapshot> &TableView::snapshots() const {
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
    for (auto &[group, snapshots] : groups) {
        draw_table_group_window(group, snapshots, group_index);
        ++group_index;
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
