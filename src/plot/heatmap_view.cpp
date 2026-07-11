#include "leakflow/plot/heatmap_view.hpp"

#include "plot_render_util.hpp"

#include <imgui.h>
#include <implot.h>

#include <cstddef>
#include <mutex>
#include <string>
#include <utility>

namespace leakflow::plot {

void HeatmapView::set_matrix(
    std::string element_name,
    std::string title,
    std::string row_label,
    std::string col_label,
    std::string value_label,
    std::vector<std::string> unit_captions,
    std::int64_t units,
    std::int64_t rows,
    std::int64_t cols,
    std::vector<double> data,
    double vmin,
    double vmax)
{
    const auto lock = std::scoped_lock(mutex_);
    HeatmapSnapshot* target = nullptr;
    for (auto& snapshot : snapshots_) {
        if (snapshot.element_name == element_name) {
            target = &snapshot;
            break;
        }
    }
    if (target == nullptr) {
        snapshots_.push_back(HeatmapSnapshot{.id = next_id_++, .element_name = std::move(element_name)});
        target = &snapshots_.back();
    }
    target->title = std::move(title);
    target->row_label = std::move(row_label);
    target->col_label = std::move(col_label);
    target->value_label = std::move(value_label);
    target->unit_captions = std::move(unit_captions);
    target->units = units;
    target->rows = rows;
    target->cols = cols;
    target->data = std::move(data);
    target->vmin = vmin;
    target->vmax = vmax;
}

const std::vector<HeatmapSnapshot>& HeatmapView::snapshots() const
{
    const auto lock = std::scoped_lock(mutex_);
    return snapshots_;
}

void HeatmapView::draw(const PlotDrawContext& /*context*/)
{
    const auto lock = std::scoped_lock(mutex_);
    auto index = 0;
    for (const auto& snapshot : snapshots_) {
        const auto cells = snapshot.units * snapshot.rows * snapshot.cols;
        if (snapshot.units <= 0 || snapshot.rows <= 0 || snapshot.cols <= 0
            || snapshot.data.size() != static_cast<std::size_t>(cells)) {
            ++index;
            continue;
        }

        const auto offset = 32.0F * static_cast<float>(index);
        ImGui::SetNextWindowPos(ImVec2(72.0F + offset, 72.0F + offset), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(660.0F, 680.0F), ImGuiCond_FirstUseEver);
        const auto window_id =
            (snapshot.title.empty() ? std::string("Heatmap") : snapshot.title) + "##heatmap_" + snapshot.element_name;
        if (ImGui::Begin(window_id.c_str())) {
            // Unit slider (only when there is more than one unit); state persists across updates.
            auto& selected = selected_unit_[snapshot.element_name];
            draw_index_slider("##unit", "unit %d", selected, static_cast<int>(snapshot.units));
            if (static_cast<std::size_t>(selected) < snapshot.unit_captions.size()
                && !snapshot.unit_captions[static_cast<std::size_t>(selected)].empty()) {
                ImGui::TextUnformatted(snapshot.unit_captions[static_cast<std::size_t>(selected)].c_str());
            } else if (!snapshot.value_label.empty()) {
                ImGui::TextUnformatted(snapshot.value_label.c_str());
            }

            const auto* matrix = snapshot.data.data() + static_cast<std::size_t>(selected) * snapshot.rows * snapshot.cols;
            ImPlot::PushColormap(ImPlotColormap_Viridis);
            const auto plot_id = std::string("##heatmap_plot_") + snapshot.element_name;
            if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-64.0F, -1.0F),
                    ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
                const auto axis_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
                // Row 0 at the top: the Y axis runs top-to-bottom.
                ImPlot::SetupAxes(snapshot.col_label.c_str(), snapshot.row_label.c_str(), axis_flags,
                    axis_flags | ImPlotAxisFlags_Invert);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, static_cast<double>(snapshot.cols), ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, static_cast<double>(snapshot.rows), ImGuiCond_Always);
                ImPlot::PlotHeatmap("##cells", matrix, static_cast<int>(snapshot.rows),
                    static_cast<int>(snapshot.cols), snapshot.vmin, snapshot.vmax, nullptr,
                    ImPlotPoint(0.0, 0.0), ImPlotPoint(static_cast<double>(snapshot.cols),
                        static_cast<double>(snapshot.rows)));
                ImPlot::EndPlot();
            }
            ImGui::SameLine();
            ImPlot::ColormapScale("##scale", snapshot.vmin, snapshot.vmax, ImVec2(56.0F, -1.0F));
            ImPlot::PopColormap();
        }
        ImGui::End();
        ++index;
    }
}

void HeatmapView::clear()
{
    const auto lock = std::scoped_lock(mutex_);
    snapshots_.clear();
    selected_unit_.clear();
    next_id_ = 1;
}

bool HeatmapView::empty() const
{
    const auto lock = std::scoped_lock(mutex_);
    return snapshots_.empty();
}

} // namespace leakflow::plot
