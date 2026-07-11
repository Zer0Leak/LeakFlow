#include "leakflow/plot/poi_table_view.hpp"

#include "plot_render_util.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace leakflow::plot {
namespace {

constexpr double neg_inf = -std::numeric_limits<double>::infinity();

// Line + highlight colors. The reference row/line and the current row/line each own a color, so
// the table's peak-cell tint matches its line in the plot below.
const ImVec4 reference_line(0.33F, 0.62F, 0.93F, 1.0F);  // blue
const ImVec4 current_line(0.95F, 0.55F, 0.22F, 1.0F);    // orange
const ImVec4 reference_highlight(0.33F, 0.62F, 0.93F, 0.34F);
const ImVec4 current_highlight(0.95F, 0.55F, 0.22F, 0.34F);

[[nodiscard]] double value_at(const std::vector<double>& values, int index)
{
    return index >= 0 && static_cast<std::size_t>(index) < values.size()
        ? values[static_cast<std::size_t>(index)]
        : std::numeric_limits<double>::quiet_NaN();
}

// The plotted / compared magnitude: signed value, or |value| when the metric toggle is on.
// NaN (absent side) stays NaN so ImPlot's SkipNaN drops it and peak-finding ignores it.
[[nodiscard]] double metric_value(double value, bool use_abs)
{
    return std::isnan(value) ? value : (use_abs ? std::abs(value) : value);
}

// Sort/compare key: like metric_value but maps NaN to -inf so absent columns sort last / never win.
[[nodiscard]] double sort_key(double value, bool use_abs)
{
    return std::isnan(value) ? neg_inf : (use_abs ? std::abs(value) : value);
}

} // namespace

void PoiTableView::set_table(
    std::string element_name,
    std::string title,
    std::string reference_label,
    std::string current_label,
    std::vector<std::int64_t> unit_ids,
    std::vector<std::string> channel_labels,
    std::vector<PoiTableGroup> groups)
{
    const auto lock = std::scoped_lock(mutex_);
    PoiTableSnapshot* target = nullptr;
    for (auto& snapshot : snapshots_) {
        if (snapshot.element_name == element_name) {
            target = &snapshot;
            break;
        }
    }
    if (target == nullptr) {
        snapshots_.push_back(PoiTableSnapshot{.id = next_id_++, .element_name = std::move(element_name)});
        target = &snapshots_.back();
    }
    target->title = std::move(title);
    target->reference_label = std::move(reference_label);
    target->current_label = std::move(current_label);
    target->unit_ids = std::move(unit_ids);
    target->channel_labels = std::move(channel_labels);
    target->groups = std::move(groups);
}

const std::vector<PoiTableSnapshot>& PoiTableView::snapshots() const
{
    const auto lock = std::scoped_lock(mutex_);
    return snapshots_;
}

void PoiTableView::draw(const PlotDrawContext& /*context*/)
{
    const auto lock = std::scoped_lock(mutex_);
    auto index = 0;
    for (const auto& snapshot : snapshots_) {
        const auto units = static_cast<int>(snapshot.unit_ids.size());
        const auto channels = static_cast<int>(snapshot.channel_labels.size());
        if (units <= 0 || channels <= 0
            || snapshot.groups.size() != static_cast<std::size_t>(units) * static_cast<std::size_t>(channels)) {
            ++index;
            continue;
        }

        const auto offset = 32.0F * static_cast<float>(index);
        ImGui::SetNextWindowPos(ImVec2(80.0F + offset, 80.0F + offset), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(760.0F, 460.0F), ImGuiCond_FirstUseEver);
        const auto window_id =
            (snapshot.title.empty() ? std::string("PoI table") : snapshot.title) + "##poitable_" + snapshot.element_name;
        if (ImGui::Begin(window_id.c_str())) {
            auto& unit = selected_unit_[snapshot.element_name];
            auto& channel = selected_channel_[snapshot.element_name];
            auto& use_abs = use_abs_[snapshot.element_name];
            auto& sort_mode = sort_mode_[snapshot.element_name];
            draw_index_slider("##unit", "unit %d", unit, units);
            draw_index_slider("##channel", "channel %d", channel, channels);
            if (channel >= 0 && static_cast<std::size_t>(channel) < snapshot.channel_labels.size()) {
                ImGui::TextUnformatted(("unit " + std::to_string(snapshot.unit_ids[static_cast<std::size_t>(unit)])
                                        + "   " + snapshot.channel_labels[static_cast<std::size_t>(channel)])
                                           .c_str());
            }

            // Metric toggle (signed vs |value|) + column sort selector; both affect what is
            // highlighted, plotted, and (for sort) the column order. State persists per element.
            ImGui::TextUnformatted("metric");
            ImGui::SameLine();
            ImGui::RadioButton("value", &use_abs, 0);
            ImGui::SameLine();
            ImGui::RadioButton("|value|", &use_abs, 1);
            ImGui::SameLine();
            ImGui::TextUnformatted("   sort");
            ImGui::SameLine();
            const char* const sort_items[] = {
                "sample", snapshot.reference_label.c_str(), snapshot.current_label.c_str()};
            ImGui::SetNextItemWidth(140.0F);
            ImGui::Combo("##sort", &sort_mode, sort_items, 3);
            const auto abs_on = use_abs != 0;

            const auto& group = snapshot.groups[static_cast<std::size_t>(unit) * static_cast<std::size_t>(channels)
                                                + static_cast<std::size_t>(channel)];
            const auto columns = static_cast<int>(group.columns.size());
            if (columns > 0) {
                // Column display order: by sample index (default), or by reference / current
                // magnitude (descending). Absent values sort last via sort_key -> -inf.
                std::vector<int> order(static_cast<std::size_t>(columns));
                std::iota(order.begin(), order.end(), 0);
                if (sort_mode == 1 || sort_mode == 2) {
                    const auto& keys = sort_mode == 1 ? group.reference_values : group.current_values;
                    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
                        return sort_key(value_at(keys, lhs), abs_on) > sort_key(value_at(keys, rhs), abs_on);
                    });
                } else {
                    std::stable_sort(order.begin(), order.end(),
                        [&](int lhs, int rhs) { return value_at(group.sample, lhs) < value_at(group.sample, rhs); });
                }

                // Per-column winner: in each column, highlight whichever row (reference / current)
                // holds the larger value (or |value|). 1 = reference, 2 = current, 0 = both absent.
                std::vector<int> winner(static_cast<std::size_t>(columns), 0);
                for (int column = 0; column < columns; ++column) {
                    const auto reference_key = sort_key(value_at(group.reference_values, column), abs_on);
                    const auto current_key = sort_key(value_at(group.current_values, column), abs_on);
                    if (reference_key == neg_inf && current_key == neg_inf) {
                        winner[static_cast<std::size_t>(column)] = 0;
                    } else {
                        winner[static_cast<std::size_t>(column)] = reference_key >= current_key ? 1 : 2;
                    }
                }

                constexpr auto flags = ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_RowBg
                    | ImGuiTableFlags_SizingFixedFit;
                const auto table_height = ImGui::GetTextLineHeightWithSpacing() * 4.5F;
                if (ImGui::BeginTable("##poitable", columns + 1, flags, ImVec2(0.0F, table_height))) {
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
                    for (int position = 0; position < columns; ++position) {
                        ImGui::TableSetupColumn(group.columns[static_cast<std::size_t>(order[position])].c_str());
                    }
                    ImGui::TableHeadersRow();

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(snapshot.reference_label.c_str());
                    for (int position = 0; position < columns; ++position) {
                        ImGui::TableNextColumn();
                        if (winner[static_cast<std::size_t>(order[position])] == 1) {
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(reference_highlight));
                        }
                        ImGui::TextUnformatted(group.reference[static_cast<std::size_t>(order[position])].c_str());
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(snapshot.current_label.c_str());
                    for (int position = 0; position < columns; ++position) {
                        ImGui::TableNextColumn();
                        if (winner[static_cast<std::size_t>(order[position])] == 2) {
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(current_highlight));
                        }
                        ImGui::TextUnformatted(group.current[static_cast<std::size_t>(order[position])].c_str());
                    }
                    ImGui::EndTable();
                }

                // Line under the table: one series per present side, in the current column order
                // (x = column position), values passed through the metric so |value| flips both.
                std::vector<double> xs(static_cast<std::size_t>(columns));
                std::iota(xs.begin(), xs.end(), 0.0);
                std::vector<double> reference_y(static_cast<std::size_t>(columns));
                std::vector<double> current_y(static_cast<std::size_t>(columns));
                for (int position = 0; position < columns; ++position) {
                    reference_y[static_cast<std::size_t>(position)] =
                        metric_value(value_at(group.reference_values, order[position]), abs_on);
                    current_y[static_cast<std::size_t>(position)] =
                        metric_value(value_at(group.current_values, order[position]), abs_on);
                }
                const auto plot_id = std::string("##poitable_plot_") + snapshot.element_name;
                if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1.0F, -1.0F))) {
                    ImPlot::SetupAxes("PoI", abs_on ? "|score|" : "score",
                        ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                    if (group.has_reference) {
                        ImPlotSpec spec;
                        spec.LineColor = reference_line;
                        spec.LineWeight = 1.8F;
                        spec.Marker = ImPlotMarker_Circle;
                        spec.MarkerSize = 3.0F;
                        spec.Flags = ImPlotLineFlags_SkipNaN;
                        ImPlot::PlotLine(snapshot.reference_label.c_str(), xs.data(), reference_y.data(), columns, spec);
                    }
                    if (group.has_current) {
                        ImPlotSpec spec;
                        spec.LineColor = current_line;
                        spec.LineWeight = 1.8F;
                        spec.Marker = ImPlotMarker_Circle;
                        spec.MarkerSize = 3.0F;
                        spec.Flags = ImPlotLineFlags_SkipNaN;
                        ImPlot::PlotLine(snapshot.current_label.c_str(), xs.data(), current_y.data(), columns, spec);
                    }
                    ImPlot::EndPlot();
                }
            }
        }
        ImGui::End();
        ++index;
    }
}

void PoiTableView::clear()
{
    const auto lock = std::scoped_lock(mutex_);
    snapshots_.clear();
    selected_unit_.clear();
    selected_channel_.clear();
    use_abs_.clear();
    sort_mode_.clear();
    next_id_ = 1;
}

bool PoiTableView::empty() const
{
    const auto lock = std::scoped_lock(mutex_);
    return snapshots_.empty();
}

} // namespace leakflow::plot
