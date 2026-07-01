#include "leakflow/plot/score_view.hpp"

#include "plot_render_util.hpp"

#include <imgui.h>
#include <implot.h>
#include <implot_internal.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plot {
namespace {

void draw_score_panel(const ScoreSnapshot &snapshot, const ScorePanel &panel, std::size_t panel_ordinal,
                      float panel_height) {
    const auto plot_id = (panel.metric.empty() ? std::string("panel") : panel.metric) + "##scorepanel_" +
                         std::to_string(snapshot.id) + "_" + std::to_string(panel_ordinal);
    if (!ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1.0F, panel_height))) {
        return;
    }
    ImPlot::SetupAxes(snapshot.x_label.c_str(), panel.y_label.c_str(), ImPlotAxisFlags_AutoFit,
                      ImPlotAxisFlags_AutoFit);

    // The ScorePlot element sets a per-unit color; the palette is only a fallback.
    static const std::array<ImVec4, 8> fallback_palette{{
        ImVec4(0.05F, 0.80F, 0.35F, 0.95F), ImVec4(0.25F, 0.55F, 1.00F, 0.95F),
        ImVec4(1.00F, 0.60F, 0.20F, 0.95F), ImVec4(0.95F, 0.35F, 0.55F, 0.95F),
        ImVec4(0.65F, 0.45F, 1.00F, 0.95F), ImVec4(0.15F, 0.85F, 0.85F, 0.95F),
        ImVec4(0.95F, 0.85F, 0.25F, 0.95F), ImVec4(0.80F, 0.80F, 0.85F, 0.95F),
    }};
    const auto series_color = [&](std::size_t index, const ScoreSeries &series) {
        return series.color ? im_color(*series.color, 0.95F) : fallback_palette[index % fallback_palette.size()];
    };

    auto *draw_list = ImPlot::GetPlotDrawList();
    const auto mouse = ImGui::GetMousePos();
    const auto plot_hovered = ImPlot::IsPlotHovered();
    const auto plot_pos = ImPlot::GetPlotPos();
    const auto plot_size = ImPlot::GetPlotSize();
    const auto plot_top = plot_pos.y;
    const auto plot_bottom = plot_pos.y + plot_size.y;
    static constexpr auto marker_radius = 4.0F;
    static constexpr auto hover_radius_squared = 9.0F * 9.0F;
    static constexpr auto vline_pixel_tolerance = 5.0F;

    // Legend visibility from the previous frame (ImPlot items persist by label). A
    // hidden series draws neither its markers nor its "latest wrong" line.
    std::vector<bool> visible(panel.series.size(), true);
    for (std::size_t index = 0; index < panel.series.size(); ++index) {
        if (const auto *item = ImPlot::GetItem(panel.series[index].label.c_str())) {
            visible[index] = item->Show;
        }
    }

    // "Latest wrong key" per visible unit: the newest point whose marker is a cross.
    // Drawn as a vertical line BEHIND the data (added to the plot draw list before
    // the series lines/markers), in the unit's color.
    struct VerticalLine {
        float pixel_x = 0.0F;
        ImVec4 color;
        std::string tooltip;
    };
    std::vector<VerticalLine> vertical_lines;
    for (std::size_t index = 0; index < panel.series.size(); ++index) {
        if (!visible[index]) {
            continue;
        }
        const auto &series = panel.series[index];
        const ScoreSeriesPoint *wrong = nullptr;
        for (const auto &point : series.points) {
            if (point.marker == TracePlotAnnotationMarker::Cross) {
                wrong = &point; // points are in increasing-x order, so the last cross is the latest
            }
        }
        if (wrong == nullptr) {
            continue;
        }
        std::string tooltip = series.label + " — latest wrong";
        for (const auto &[key, value] : wrong->fields) {
            tooltip += "\n" + key + ": " + value;
        }
        vertical_lines.push_back(VerticalLine{
            .pixel_x = ImPlot::PlotToPixels(wrong->x, 0.0).x,
            .color = series_color(index, series),
            .tooltip = std::move(tooltip),
        });
    }

    for (const auto &line : vertical_lines) {
        const auto faint = ImGui::GetColorU32(ImVec4(line.color.x, line.color.y, line.color.z, 0.40F));
        draw_list->AddLine(ImVec2(line.pixel_x, plot_top), ImVec2(line.pixel_x, plot_bottom), faint, 2.0F);
    }

    // Count badge where several lines pile up on screen (zoom in to separate them).
    {
        std::vector<float> pixels;
        pixels.reserve(vertical_lines.size());
        for (const auto &line : vertical_lines) {
            pixels.push_back(line.pixel_x);
        }
        std::sort(pixels.begin(), pixels.end());
        std::size_t start = 0;
        while (start < pixels.size()) {
            std::size_t end = start + 1;
            while (end < pixels.size() && pixels[end] - pixels[end - 1] <= vline_pixel_tolerance) {
                ++end;
            }
            if (end - start > 1) {
                const auto mid = (pixels[start] + pixels[end - 1]) * 0.5F;
                draw_annotation_number_label(*draw_list, ImVec2(mid, plot_top + 10.0F), std::to_string(end - start));
            }
            start = end;
        }
    }

    // Vertical-line hover: every line within tolerance of the cursor (shows "both").
    std::optional<std::string> vline_tooltip;
    if (plot_hovered && mouse.y >= plot_top && mouse.y <= plot_bottom) {
        std::string combined;
        for (const auto &line : vertical_lines) {
            if (std::abs(mouse.x - line.pixel_x) <= vline_pixel_tolerance) {
                if (!combined.empty()) {
                    combined += "\n\n";
                }
                combined += line.tooltip;
            }
        }
        if (!combined.empty()) {
            vline_tooltip = std::move(combined);
        }
    }

    // Series lines (PlotLine self-hides on a legend toggle) + markers, gated on
    // visibility so a hidden series also drops its manually-drawn markers.
    std::optional<std::pair<float, std::string>> marker_hover;
    for (std::size_t index = 0; index < panel.series.size(); ++index) {
        const auto &series = panel.series[index];
        if (series.points.empty()) {
            continue;
        }
        if (series.secondary && !snapshot.show_secondary) {
            continue; // secondary series hidden by the show_second_score property
        }
        std::vector<double> xs;
        std::vector<double> ys;
        xs.reserve(series.points.size());
        ys.reserve(series.points.size());
        for (const auto &point : series.points) {
            xs.push_back(point.x);
            ys.push_back(point.y);
        }
        const auto color = series_color(index, series);
        ImPlotSpec spec;
        // A secondary series (e.g. second-best score) shares the primary's label so a
        // single legend entry toggles both; it is drawn fainter/thinner, no markers.
        spec.LineColor = series.secondary ? ImVec4(color.x, color.y, color.z, 0.50F) : color;
        spec.LineWeight = series.secondary ? 1.0F : 1.6F;
        ImPlot::PlotLine(series.label.c_str(), xs.data(), ys.data(), static_cast<int>(xs.size()), spec);
        if (!visible[index] || series.secondary) {
            continue;
        }
        for (const auto &point : series.points) {
            const auto pixel = ImPlot::PlotToPixels(point.x, point.y);
            draw_annotation_marker(*draw_list, pixel, marker_radius, {color}, point.marker);
            if (plot_hovered) {
                const auto distance = squared_distance(mouse, pixel);
                if (distance <= hover_radius_squared && (!marker_hover || distance < marker_hover->first)) {
                    std::string tooltip = series.label;
                    for (const auto &[key, value] : point.fields) {
                        tooltip += "\n" + key + ": " + value;
                    }
                    marker_hover = std::make_pair(distance, std::move(tooltip));
                }
            }
        }
    }

    // A marker (specific point) wins over a vertical line (broad) when both hovered.
    if (marker_hover) {
        ImGui::SetTooltip("%s", marker_hover->second.c_str());
    } else if (vline_tooltip) {
        ImGui::SetTooltip("%s", vline_tooltip->c_str());
    }
    ImPlot::EndPlot();
}

void draw_score_group_window(std::string_view group, const std::vector<const ScoreSnapshot *> &snapshots,
                             int group_index, std::map<std::string, float> &panel_heights) {
    std::string title;
    for (const auto *snapshot : snapshots) {
        if (!snapshot->title.empty()) {
            title = snapshot->title;
            break;
        }
    }
    if (title.empty()) {
        title = group.empty() || group == "default" ? "Scores" : std::string(group);
    }

    const auto window_id = title + "##scoreplot_group_" + std::string(group);
    ImGui::SetNextWindowPos(ImVec2(48.0F + 36.0F * static_cast<float>(group_index),
                                   48.0F + 36.0F * static_cast<float>(group_index)),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(760.0F, 620.0F), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(window_id.c_str())) {
        const auto group_text = "group: " + std::string(group);
        ImGui::TextUnformatted(group_text.c_str());
        std::size_t panel_ordinal = 0;
        for (const auto *snapshot : snapshots) {
            for (const auto &panel : snapshot->panels) {
                if (panel_ordinal != 0) {
                    ImGui::Spacing();
                }
                const auto panel_key =
                    std::to_string(snapshot->id) + "/" + std::to_string(panel_ordinal) + "/" + panel.metric;
                auto &panel_height = panel_heights.try_emplace(panel_key, 200.0F).first->second;
                draw_score_panel(*snapshot, panel, panel_ordinal, panel_height);

                // Draggable splitter to resize this panel taller/shorter.
                ImGui::InvisibleButton(("##score_splitter_" + panel_key).c_str(), ImVec2(-1.0F, 7.0F));
                const auto splitter_active = ImGui::IsItemActive();
                if (splitter_active) {
                    panel_height = std::clamp(panel_height + ImGui::GetIO().MouseDelta.y, 80.0F, 1600.0F);
                }
                if (splitter_active || ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                }
                const auto grip_min = ImGui::GetItemRectMin();
                const auto grip_max = ImGui::GetItemRectMax();
                const auto grip_mid_y = (grip_min.y + grip_max.y) * 0.5F;
                const auto grip_color = ImGui::GetColorU32(
                    splitter_active ? ImGuiCol_SeparatorActive
                                    : (ImGui::IsItemHovered() ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator));
                ImGui::GetWindowDrawList()->AddLine(ImVec2(grip_min.x + 24.0F, grip_mid_y),
                                                    ImVec2(grip_max.x - 24.0F, grip_mid_y), grip_color, 2.0F);
                ++panel_ordinal;
            }
        }
    }
    ImGui::End();
}

} // namespace

void ScoreView::append_points(std::string element_name, std::string group, std::string title, std::string x_label,
                              bool show_secondary, const std::vector<ScorePointUpdate> &updates) {
    const auto lock = std::scoped_lock(mutex_);
    if (group.empty()) {
        group = "default";
    }

    ScoreSnapshot *snapshot = nullptr;
    for (auto &existing : snapshots_) {
        if (existing.element_name == element_name) {
            snapshot = &existing;
            break;
        }
    }
    if (snapshot == nullptr) {
        ScoreSnapshot fresh;
        fresh.id = next_id_++;
        fresh.element_name = std::move(element_name);
        snapshot = &snapshots_.emplace_back(std::move(fresh));
    }
    // Presentation is refreshed each call (like a ui-control change).
    snapshot->group = std::move(group);
    snapshot->title = std::move(title);
    snapshot->x_label = std::move(x_label);
    snapshot->show_secondary = show_secondary;

    for (const auto &update : updates) {
        ScorePanel *panel = nullptr;
        for (auto &candidate : snapshot->panels) {
            if (candidate.metric == update.panel) {
                panel = &candidate;
                break;
            }
        }
        if (panel == nullptr) {
            panel = &snapshot->panels.emplace_back(ScorePanel{.metric = update.panel, .y_label = update.panel_y_label});
        }

        // A primary and a secondary series can share a label; keep them separate.
        ScoreSeries *series = nullptr;
        for (auto &candidate : panel->series) {
            if (candidate.label == update.series && candidate.secondary == update.secondary) {
                series = &candidate;
                break;
            }
        }
        if (series == nullptr) {
            series = &panel->series.emplace_back(
                ScoreSeries{.label = update.series, .color = update.color, .secondary = update.secondary});
        }
        series->points.push_back(update.point);
    }
}

void ScoreView::set_show_secondary(std::string_view element_name, bool show_secondary) {
    const auto lock = std::scoped_lock(mutex_);
    for (auto &snapshot : snapshots_) {
        if (snapshot.element_name == element_name) {
            snapshot.show_secondary = show_secondary;
            return;
        }
    }
}

const std::vector<ScoreSnapshot> &ScoreView::snapshots() const {
    const auto lock = std::scoped_lock(mutex_);
    return snapshots_;
}

void ScoreView::draw() {
    // Called on the UI thread; lock our own data against the worker's append_points.
    const auto lock = std::scoped_lock(mutex_);
    std::map<std::string, std::vector<const ScoreSnapshot *>> groups;
    for (const auto &snapshot : snapshots_) {
        groups[snapshot.group].push_back(&snapshot);
    }
    auto group_index = 0;
    for (const auto &[group, snapshots] : groups) {
        draw_score_group_window(group, snapshots, group_index, panel_heights_);
        ++group_index;
    }
}

void ScoreView::clear() {
    const auto lock = std::scoped_lock(mutex_);
    snapshots_.clear();
    panel_heights_.clear();
    next_id_ = 1;
}

bool ScoreView::empty() const {
    const auto lock = std::scoped_lock(mutex_);
    return snapshots_.empty();
}

} // namespace leakflow::plot
