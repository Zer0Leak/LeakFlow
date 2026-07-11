#include "plot_render_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace leakflow::plot {

int draw_index_slider(const char *id, const char *label_format, int &value, int count) {
    if (count <= 0) {
        return 0;
    }
    if (value >= count) {
        value = 0;
    }
    if (count > 1) {
        ImGui::SetNextItemWidth(-1.0F);
        ImGui::SliderInt(id, &value, 0, count - 1, label_format);
        value = std::clamp(value, 0, count - 1);
    }
    return value;
}

namespace {

[[nodiscard]] bool colors_are_close(ImVec4 left, ImVec4 right) {
    static constexpr auto epsilon = 1.0e-4F;
    return std::abs(left.x - right.x) <= epsilon && std::abs(left.y - right.y) <= epsilon &&
           std::abs(left.z - right.z) <= epsilon && std::abs(left.w - right.w) <= epsilon;
}

[[nodiscard]] bool colors_are_uniform(const std::vector<ImVec4> &colors) {
    return std::all_of(colors.begin(), colors.end(),
                       [&colors](auto color) { return colors.empty() || colors_are_close(color, colors.front()); });
}

} // namespace

ImVec4 im_color(const TracePlotColor &color, float alpha) {
    return ImVec4(color.red, color.green, color.blue, alpha);
}

float squared_distance(ImVec2 left, ImVec2 right) {
    const auto dx = left.x - right.x;
    const auto dy = left.y - right.y;
    return dx * dx + dy * dy;
}

void draw_annotation_number_label(ImDrawList &draw_list, ImVec2 marker, const std::string &text) {
    if (text.empty()) {
        return;
    }
    const auto text_position = ImVec2(marker.x + 5.0F, marker.y - 8.0F);
    draw_list.AddText(ImVec2(text_position.x + 1.0F, text_position.y + 1.0F),
                      ImGui::GetColorU32(ImVec4(0.0F, 0.0F, 0.0F, 0.85F)), text.c_str());
    draw_list.AddText(text_position, ImGui::GetColorU32(ImVec4(1.0F, 1.0F, 1.0F, 0.95F)), text.c_str());
}

void draw_annotation_marker(ImDrawList &draw_list, ImVec2 marker, float marker_radius,
                            const std::vector<ImVec4> &colors, TracePlotAnnotationMarker shape) {
    if (colors.empty()) {
        return;
    }

    const auto outline = ImGui::GetColorU32(ImVec4(0.0F, 0.0F, 0.0F, 0.75F));

    // A single-color marker draws its shape; a multi-color group (overlapping
    // annotations) always draws the circular pie so the colors blend cleanly.
    if (colors.size() == 1 || colors_are_uniform(colors)) {
        const auto color = ImGui::GetColorU32(colors.front());
        if (shape == TracePlotAnnotationMarker::Square) {
            const ImVec2 top_left(marker.x - marker_radius, marker.y - marker_radius);
            const ImVec2 bottom_right(marker.x + marker_radius, marker.y + marker_radius);
            draw_list.AddRectFilled(top_left, bottom_right, color);
            draw_list.AddRect(top_left, bottom_right, outline, 0.0F, 0, 1.0F);
            return;
        }
        if (shape == TracePlotAnnotationMarker::Cross) {
            const auto thickness = std::max(2.0F, marker_radius * 0.7F);
            const ImVec2 top_left(marker.x - marker_radius, marker.y - marker_radius);
            const ImVec2 bottom_right(marker.x + marker_radius, marker.y + marker_radius);
            const ImVec2 top_right(marker.x + marker_radius, marker.y - marker_radius);
            const ImVec2 bottom_left(marker.x - marker_radius, marker.y + marker_radius);
            draw_list.AddLine(top_left, bottom_right, outline, thickness + 1.5F);
            draw_list.AddLine(bottom_left, top_right, outline, thickness + 1.5F);
            draw_list.AddLine(top_left, bottom_right, color, thickness);
            draw_list.AddLine(bottom_left, top_right, color, thickness);
            return;
        }
        draw_list.AddCircleFilled(marker, marker_radius, color, 12);
        draw_list.AddCircle(marker, marker_radius + 1.0F, outline, 12, 1.0F);
        return;
    }

    static constexpr auto pi = 3.14159265358979323846F;
    const auto slice_angle = 2.0F * pi / static_cast<float>(colors.size());
    for (std::size_t index = 0; index < colors.size(); ++index) {
        const auto start = -0.5F * pi + slice_angle * static_cast<float>(index);
        const auto end = start + slice_angle;
        draw_list.PathClear();
        draw_list.PathLineTo(marker);
        draw_list.PathArcTo(marker, marker_radius, start, end, 8);
        draw_list.PathFillConvex(ImGui::GetColorU32(colors[index]));
    }
    draw_list.AddCircle(marker, marker_radius + 1.0F, outline, 12, 1.0F);
}

} // namespace leakflow::plot
