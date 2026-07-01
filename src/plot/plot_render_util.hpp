#pragma once

#include "leakflow/plot/plot_runtime.hpp"

#include <imgui.h>

#include <string>
#include <vector>

// Internal (leakflow_plot) drawing primitives shared by the plot views (TraceView,
// ScoreView, ...). Domain-free: they know colors, marker shapes, and pixels -- not
// traces or scores. Keeping them here lets each view own its own rendering TU while
// reusing the same marker/label helpers.
namespace leakflow::plot {

[[nodiscard]] ImVec4 im_color(const TracePlotColor &color, float alpha);
[[nodiscard]] float squared_distance(ImVec2 left, ImVec2 right);
void draw_annotation_number_label(ImDrawList &draw_list, ImVec2 marker, const std::string &text);
void draw_annotation_marker(ImDrawList &draw_list, ImVec2 marker, float marker_radius,
                            const std::vector<ImVec4> &colors, TracePlotAnnotationMarker shape);

} // namespace leakflow::plot
