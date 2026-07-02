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

void draw_table_snapshot(TableSnapshot &snapshot) {
    if (snapshot.frames.empty() || snapshot.columns.empty()) {
        return;
    }
    if (!snapshot.title.empty()) {
        ImGui::SeparatorText(snapshot.title.c_str());
    }

    // History scrubber: -1 follows the latest frame; the slider (1-based) parks at the
    // end to keep following, otherwise pins the chosen frame.
    const auto frame_count = static_cast<int>(snapshot.frames.size());
    auto frame_index = (snapshot.cursor < 0 || snapshot.cursor >= frame_count) ? frame_count - 1 : snapshot.cursor;
    if (frame_count > 1) {
        auto display = frame_index + 1;
        ImGui::PushID(static_cast<int>(snapshot.id));
        ImGui::SetNextItemWidth(220.0F);
        if (ImGui::SliderInt("history", &display, 1, frame_count)) {
            snapshot.cursor = display == frame_count ? -1 : display - 1;
        }
        frame_index = display - 1;
        ImGui::PopID();
    }
    const auto &frame = snapshot.frames[static_cast<std::size_t>(frame_index)];
    if (!frame.caption.empty()) {
        if (frame_count > 1) {
            ImGui::SameLine();
        }
        ImGui::TextUnformatted(frame.caption.c_str());
    }

    const auto column_count = static_cast<int>(snapshot.columns.size());
    const auto visible_rows = std::min<std::size_t>(frame.rows.size() + 1, 26);
    const auto table_height = static_cast<float>(visible_rows) * ImGui::GetTextLineHeightWithSpacing() + 8.0F;
    const auto table_id = "##scoretable_" + std::to_string(snapshot.id);
    constexpr auto flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable(table_id.c_str(), column_count, flags, ImVec2(0.0F, table_height))) {
        return;
    }
    // Freeze the header row and the first (rank) column while scrolling.
    ImGui::TableSetupScrollFreeze(1, 1);
    for (const auto &header : snapshot.columns) {
        ImGui::TableSetupColumn(header.c_str());
    }
    ImGui::TableHeadersRow();

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
            if (cell.emphasize) {
                ImGui::TextColored(ImVec4(1.0F, 1.0F, 1.0F, 1.0F), "%s", cell.text.c_str());
            } else {
                ImGui::TextUnformatted(cell.text.c_str());
            }
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
        for (std::size_t index = 0; index < snapshots.size(); ++index) {
            if (index != 0) {
                ImGui::Spacing();
            }
            draw_table_snapshot(*snapshots[index]);
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
