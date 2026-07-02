#include "leakflow/plot/trace_view.hpp"

#include "leakflow/log/logger.hpp"
#include "leakflow/plot/pipeline_graph.hpp"
#include "plot_render_util.hpp"

#include <imgui.h>
#include <implot.h>
#include <implot_internal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plot {
namespace {

[[nodiscard]] TracePlotColor default_trace_color(std::uint64_t id) {
    static constexpr std::array<TracePlotColor, 8> palette{{
        TracePlotColor{0.05F, 0.80F, 0.35F},
        TracePlotColor{0.25F, 0.55F, 1.00F},
        TracePlotColor{1.00F, 0.60F, 0.20F},
        TracePlotColor{0.95F, 0.35F, 0.55F},
        TracePlotColor{0.65F, 0.45F, 1.00F},
        TracePlotColor{0.15F, 0.85F, 0.85F},
        TracePlotColor{0.95F, 0.85F, 0.25F},
        TracePlotColor{0.80F, 0.80F, 0.85F},
    }};

    return palette[static_cast<std::size_t>((id - 1U) % palette.size())];
}
[[nodiscard]] bool color_component_is_valid(float value) { return value >= 0.0F && value <= 1.0F; }
[[nodiscard]] bool color_is_valid(const TracePlotColor &color) {
    return color_component_is_valid(color.red) && color_component_is_valid(color.green) &&
           color_component_is_valid(color.blue);
}
void validate_snapshot(const TracePlotSnapshot &snapshot) {
    if (snapshot.rank() != 1 && snapshot.rank() != 2) {
        throw std::invalid_argument("TracePlotSnapshot requires rank 1 or rank 2 data");
    }
    if (snapshot.sample_count() <= 0) {
        throw std::invalid_argument("TracePlotSnapshot requires at least one sample");
    }
    if (snapshot.trace_count() <= 0) {
        throw std::invalid_argument("TracePlotSnapshot requires at least one trace");
    }

    const auto expected = snapshot.trace_count() * snapshot.sample_count();
    if (expected < 0 ||
        static_cast<std::uint64_t>(expected) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("TracePlotSnapshot shape is too large");
    }
    if (snapshot.values.size() != static_cast<std::size_t>(expected)) {
        throw std::invalid_argument("TracePlotSnapshot values do not match shape");
    }
    if (snapshot.initial_trace_index < 0 || snapshot.initial_trace_index >= snapshot.trace_count()) {
        throw std::invalid_argument("TracePlotSnapshot initial trace index is out of range");
    }
    if (snapshot.alpha < 0.0 || snapshot.alpha > 1.0) {
        throw std::invalid_argument("TracePlotSnapshot alpha must be between 0 and 1");
    }
    if (snapshot.line_width <= 0.0) {
        throw std::invalid_argument("TracePlotSnapshot line width must be positive");
    }
    if (snapshot.color && !color_is_valid(*snapshot.color)) {
        throw std::invalid_argument("TracePlotSnapshot color components must be between 0 and 1");
    }
    if (snapshot.sample_rate_hz && *snapshot.sample_rate_hz <= 0.0) {
        throw std::invalid_argument("TracePlotSnapshot sample rate must be positive when set");
    }
    for (const auto &annotation : snapshot.annotations) {
        if (annotation.sample_index < 0 || annotation.sample_index >= snapshot.sample_count()) {
            throw std::invalid_argument("TracePlotSnapshot annotation sample index is out of range");
        }
        if (annotation.value && !std::isfinite(*annotation.value)) {
            throw std::invalid_argument("TracePlotSnapshot annotation value must be finite when present");
        }
        if (annotation.norm_value &&
            (!std::isfinite(*annotation.norm_value) || *annotation.norm_value < -1.0 || *annotation.norm_value > 1.0)) {
            throw std::invalid_argument("TracePlotSnapshot annotation norm_value "
                                        "must be finite and between -1 and 1");
        }
        if (annotation.kind.empty()) {
            throw std::invalid_argument("TracePlotSnapshot annotation kind cannot be empty");
        }
        for (const auto &[key, value] : annotation.fields) {
            (void)value;
            if (key.empty()) {
                throw std::invalid_argument("TracePlotSnapshot annotation field keys cannot be empty");
            }
        }
    }
}

struct TracePanel {
    std::vector<const TracePlotSnapshot *> snapshots;
};

[[nodiscard]] std::string snapshot_label(const TracePlotSnapshot &snapshot) {
    if (!snapshot.label.empty()) {
        return snapshot.label;
    }
    return "trace";
}

[[nodiscard]] std::string snapshot_display_name(const TracePlotSnapshot &snapshot) {
    const auto label = snapshot_label(snapshot);
    if (!snapshot.element_name.empty() && snapshot.element_name != label) {
        return label + " (" + snapshot.element_name + ")";
    }

    return label;
}

[[nodiscard]] std::string group_title(std::string_view group, const std::vector<const TracePlotSnapshot *> &snapshots) {
    for (const auto *snapshot : snapshots) {
        if (!snapshot->title.empty()) {
            return snapshot->title;
        }
    }

    if (!group.empty() && group != "default") {
        return std::string(group);
    }

    return "TracePlot";
}

[[nodiscard]] std::string x_axis_label(const TracePlotSnapshot &snapshot) {
    if (!snapshot.x_label.empty()) {
        return snapshot.x_label;
    }
    if (snapshot.x_axis == TracePlotXAxis::TimeUs && snapshot.sample_rate_hz) {
        return "time (us)";
    }
    return "sample";
}

[[nodiscard]] std::string y_axis_label(const TracePlotSnapshot &snapshot) {
    if (!snapshot.y_label.empty()) {
        return snapshot.y_label;
    }
    return "value";
}

[[nodiscard]] std::vector<const TracePlotSnapshot *>
ordered_snapshots(TraceView &view, const std::vector<const TracePlotSnapshot *> &snapshots) {
    auto ordered = snapshots;
    std::stable_sort(ordered.begin(), ordered.end(), [&view](const auto *left, const auto *right) {
        const auto &left_state = view.mutable_trace_display_state(*left);
        const auto &right_state = view.mutable_trace_display_state(*right);
        if (left_state.order != right_state.order) {
            return left_state.order < right_state.order;
        }

        return left->id < right->id;
    });

    return ordered;
}

void apply_display_order(TraceView &view, const std::vector<const TracePlotSnapshot *> &snapshots) {
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
        view.mutable_trace_display_state(*snapshots[index]).order = static_cast<std::int64_t>(index);
    }
}

void move_snapshot_before(TraceView &view, std::vector<const TracePlotSnapshot *> snapshots,
                          std::uint64_t dragged_id, std::uint64_t target_id) {
    if (dragged_id == target_id) {
        return;
    }

    const auto matches_id = [](std::uint64_t id) {
        return [id](const TracePlotSnapshot *snapshot) { return snapshot->id == id; };
    };

    const auto dragged = std::find_if(snapshots.begin(), snapshots.end(), matches_id(dragged_id));
    if (dragged == snapshots.end()) {
        return;
    }

    const auto *snapshot = *dragged;
    snapshots.erase(dragged);

    const auto target = std::find_if(snapshots.begin(), snapshots.end(), matches_id(target_id));
    snapshots.insert(target, snapshot);
    apply_display_order(view, snapshots);
}

[[nodiscard]] std::vector<TracePanel> build_trace_panels(const std::vector<const TracePlotSnapshot *> &snapshots) {
    std::vector<TracePanel> panels;
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
        const auto *snapshot = snapshots[index];
        if (panels.empty() || (index != 0 && snapshot->layout == TracePlotLayout::Stacked)) {
            panels.push_back({});
        }

        panels.back().snapshots.push_back(snapshot);
    }

    return panels;
}

[[nodiscard]] int clamped_trace_index(const TracePlotSnapshot &snapshot, int value) {
    return std::clamp(value, 0, static_cast<int>(snapshot.trace_count() - 1));
}

[[nodiscard]] int selected_trace_index(TraceView &view, const TracePlotSnapshot &snapshot) {
    if (snapshot.rank() == 1) {
        return 0;
    }

    auto &trace_index = view.mutable_trace_index(snapshot.id, static_cast<int>(snapshot.initial_trace_index));
    // An accumulate snapshot streams new traces in; while streaming, the slider rides
    // the newest one by default. The slider widget clears follow_latest while the user
    // holds it; once a run is held (Idle) or frozen (Paused) we stop following so the
    // history can be scrubbed freely.
    if (snapshot.accumulate && view.streaming() && view.mutable_trace_follow_latest(snapshot.id, true)) {
        trace_index = static_cast<int>(snapshot.trace_count() - 1);
    }
    trace_index = clamped_trace_index(snapshot, trace_index);
    return trace_index;
}

[[nodiscard]] bool snapshot_has_trace_slider(const TracePlotSnapshot &snapshot) {
    return snapshot.rank() == 2;
}

[[nodiscard]] bool has_trace_sliders(const std::vector<const TracePlotSnapshot *> &snapshots) {
    return std::any_of(snapshots.begin(), snapshots.end(),
                       [](const auto *snapshot) { return snapshot_has_trace_slider(*snapshot); });
}

[[nodiscard]] bool default_group_trace_lock(const std::vector<const TracePlotSnapshot *> &snapshots) {
    return std::any_of(snapshots.begin(), snapshots.end(), [](const auto *snapshot) {
        return snapshot_has_trace_slider(*snapshot) && snapshot->lock_trace_index;
    });
}

[[nodiscard]] int trace_control_count(const std::vector<const TracePlotSnapshot *> &snapshots) {
    auto count = 0;

    for (const auto *snapshot : snapshots) {
        if (snapshot_has_trace_slider(*snapshot)) {
            ++count;
        }
    }

    return count;
}

bool draw_vertical_slider(std::string_view tooltip_label, const std::string &imgui_id, int &value, int max_value,
                          float height, const TracePlotColor &color, float alpha) {
    const auto previous_value = std::clamp(value, 0, max_value);
    auto display_value = max_value - std::clamp(value, 0, max_value);

    const auto changed = ImGui::VSliderInt(imgui_id.c_str(), ImVec2(42.0F, height), &display_value, 0, max_value, "");
    value = max_value - std::clamp(display_value, 0, max_value);
    // Traces are presented to the user as 1..N; the stored value stays the 0-based index.
    const auto value_text = std::to_string(value + 1);
    const auto item_min = ImGui::GetItemRectMin();
    const auto item_max = ImGui::GetItemRectMax();
    const auto text_size = ImGui::CalcTextSize(value_text.c_str());
    const auto text_position = ImVec2(item_min.x + std::max(0.0F, (item_max.x - item_min.x - text_size.x) * 0.5F),
                                      item_min.y + std::max(0.0F, (item_max.y - item_min.y - text_size.y) * 0.5F));
    auto *draw_list = ImGui::GetWindowDrawList();
    const auto accent_alpha = std::clamp(alpha, 0.45F, 0.90F);
    draw_list->AddRectFilled(ImVec2(item_min.x + 3.0F, item_min.y + 4.0F), ImVec2(item_min.x + 6.0F, item_max.y - 4.0F),
                             ImGui::GetColorU32(im_color(color, accent_alpha)), 2.0F);
    draw_list->AddText(ImVec2(text_position.x + 1.0F, text_position.y + 1.0F),
                       ImGui::GetColorU32(ImVec4(0.0F, 0.0F, 0.0F, 0.85F)), value_text.c_str());
    draw_list->AddText(text_position, ImGui::GetColorU32(ImVec4(1.0F, 1.0F, 1.0F, 0.95F)), value_text.c_str());
    if (ImGui::IsItemHovered()) {
        const auto tooltip = std::string(tooltip_label) + "\ntrace " + std::to_string(value + 1) + " / " +
                             std::to_string(max_value + 1);
        ImGui::SetTooltip("%s", tooltip.c_str());
    }
    return changed && value != previous_value;
}

[[nodiscard]] int clamped_group_trace_delta(TraceView &view,
                                            const std::vector<const TracePlotSnapshot *> &snapshots,
                                            int requested_delta) {
    if (requested_delta == 0) {
        return 0;
    }

    auto lower_delta = std::numeric_limits<int>::min();
    auto upper_delta = std::numeric_limits<int>::max();
    auto saw_trace = false;

    for (const auto *snapshot : snapshots) {
        if (!snapshot_has_trace_slider(*snapshot)) {
            continue;
        }

        auto &trace_index = view.mutable_trace_index(snapshot->id, static_cast<int>(snapshot->initial_trace_index));
        trace_index = clamped_trace_index(*snapshot, trace_index);
        const auto max_trace_index = static_cast<int>(snapshot->trace_count() - 1);
        lower_delta = std::max(lower_delta, -trace_index);
        upper_delta = std::min(upper_delta, max_trace_index - trace_index);
        saw_trace = true;
    }

    if (!saw_trace || lower_delta > upper_delta) {
        return 0;
    }
    return std::clamp(requested_delta, lower_delta, upper_delta);
}

void apply_group_trace_delta(TraceView &view,
                             const std::vector<const TracePlotSnapshot *> &snapshots,
                             int delta) {
    if (delta == 0) {
        return;
    }

    for (const auto *snapshot : snapshots) {
        if (!snapshot_has_trace_slider(*snapshot)) {
            continue;
        }

        auto &trace_index = view.mutable_trace_index(snapshot->id, static_cast<int>(snapshot->initial_trace_index));
        trace_index = clamped_trace_index(*snapshot, trace_index + delta);
    }
}

void draw_vertical_trace_controls(TraceView &view, std::string_view group,
                                  const std::vector<const TracePlotSnapshot *> &snapshots,
                                  const std::vector<const TracePlotSnapshot *> &group_snapshots,
                                  float plot_height) {
    const auto slider_height = std::max(120.0F, plot_height);
    const auto group_has_multiple_sliders = trace_control_count(group_snapshots) > 1;
    auto drew_control = false;
    auto &group_trace_lock = view.mutable_group_trace_lock(group, default_group_trace_lock(group_snapshots));

    for (const auto *snapshot : snapshots) {
        if (!snapshot_has_trace_slider(*snapshot)) {
            continue;
        }

        if (drew_control) {
            ImGui::SameLine();
        }

        auto &trace_index = view.mutable_trace_index(snapshot->id, static_cast<int>(snapshot->initial_trace_index));
        trace_index = clamped_trace_index(*snapshot, trace_index);
        auto requested_trace_index = trace_index;
        const auto &display_state = view.mutable_trace_display_state(*snapshot);
        const auto changed =
            draw_vertical_slider(snapshot_display_name(*snapshot), "##trace_index_" + std::to_string(snapshot->id),
                                 requested_trace_index, static_cast<int>(snapshot->trace_count() - 1), slider_height,
                                 display_state.color, display_state.alpha);
        // Captured immediately after the slider so it reflects the slider widget.
        const auto slider_active = ImGui::IsItemActive();

        // Right-clicking any slider exposes the group trace-lock toggle, but only when
        // the group has more than one trace slider to keep in sync. group_snapshots
        // spans the whole group, so this behaves the same in stacked and overlay layouts.
        if (group_has_multiple_sliders) {
            const auto popup_id = "##trace_lock_ctx_" + std::to_string(snapshot->id);
            if (ImGui::BeginPopupContextItem(popup_id.c_str())) {
                ImGui::Checkbox("Lock trace sliders", &group_trace_lock);
                ImGui::EndPopup();
            }
        }

        if (changed) {
            if (group_trace_lock) {
                const auto delta = clamped_group_trace_delta(view, group_snapshots, requested_trace_index - trace_index);
                apply_group_trace_delta(view, group_snapshots, delta);
                // Push each affected slider back to its element's trace_index property.
                // Accumulate sliders scrub view history only, so they are not
                // written back (it would push an out-of-range index onto the next
                // single-buffer snapshot).
                for (const auto *grouped : group_snapshots) {
                    if (grouped->accumulate) {
                        continue;
                    }
                    view.notify_trace_index(
                        grouped->element_name,
                        view.mutable_trace_index(grouped->id, static_cast<int>(grouped->initial_trace_index)));
                }
            } else {
                trace_index = requested_trace_index;
                if (!snapshot->accumulate) {
                    view.notify_trace_index(snapshot->element_name, trace_index);
                }
            }
        }

        // Accumulate plots ride the newest trace; holding the slider (or a group
        // trace lock owning the index) pauses the follow so a past trace can be
        // inspected. On release it resumes and snaps back to the latest.
        if (snapshot->accumulate) {
            view.mutable_trace_follow_latest(snapshot->id, true) = !group_trace_lock && !slider_active;
        }
        drew_control = true;
    }
}

// All traces in an overlay sub-group share one x-axis: the front snapshot of the
// panel is the axis source, so every overlaid trace is plotted on the same scale.
[[nodiscard]] std::vector<float> x_values_for(const TracePlotSnapshot &snapshot,
                                              const TracePlotSnapshot &axis_source) {
    std::vector<float> x_values;
    if (axis_source.x_axis != TracePlotXAxis::TimeUs || !axis_source.sample_rate_hz) {
        return x_values;
    }

    const auto count = snapshot.sample_count();
    x_values.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        x_values.push_back(static_cast<float>(static_cast<double>(index) * 1'000'000.0 / *axis_source.sample_rate_hz));
    }

    return x_values;
}

[[nodiscard]] double annotation_x_value(const TracePlotSnapshot &snapshot, std::int64_t sample_index,
                                        const std::vector<float> &x_values) {
    if (!x_values.empty()) {
        return x_values.at(static_cast<std::size_t>(sample_index));
    }

    return static_cast<double>(sample_index);
}

[[nodiscard]] std::string annotation_text_for(const TracePlotAnnotation &annotation) {
    if (!annotation.fields.empty()) {
        std::string text;
        for (const auto &[key, value] : annotation.fields) {
            if (!text.empty()) {
                text += ", ";
            }
            text += key + "=" + value;
        }
        return text;
    }
    if (!annotation.text.empty()) {
        return annotation.text;
    }
    if (!annotation.label.empty()) {
        return annotation.label;
    }

    return annotation.kind;
}

[[nodiscard]] std::string annotation_legend_label_for(const TracePlotAnnotation &annotation) {
    for (const auto &[key, value] : annotation.fields) {
        if (key == "label" && !value.empty()) {
            return value;
        }
    }
    if (!annotation.label.empty()) {
        return annotation.label;
    }
    return annotation.kind;
}

[[nodiscard]] std::string annotation_tooltip_for(const TracePlotSnapshot &snapshot,
                                                 const TracePlotAnnotation &annotation, double x_value) {
    auto tooltip = annotation_text_for(annotation);
    if (!annotation.fields.empty()) {
        tooltip.clear();
        for (const auto &[key, value] : annotation.fields) {
            if (!tooltip.empty()) {
                tooltip += "\n";
            }
            tooltip += key + ": " + value;
        }
    }
    tooltip += "\nsample_index: " + std::to_string(annotation.sample_index);
    if (annotation.value) {
        tooltip += "\nvalue: " + std::to_string(*annotation.value);
    } else if (annotation.norm_value) {
        tooltip += "\nnorm_value: " + std::to_string(*annotation.norm_value);
    } else {
        tooltip += "\nposition: top";
    }
    if (snapshot.x_axis == TracePlotXAxis::TimeUs && snapshot.sample_rate_hz) {
        tooltip += "\ntime_us: " + std::to_string(x_value);
    }
    tooltip += "\nkind: " + annotation.kind;
    if (annotation.target_index) {
        tooltip += "\ntarget_index: " + std::to_string(*annotation.target_index);
    }
    return tooltip;
}

[[nodiscard]] ImVec4 annotation_marker_color(const TracePlotAnnotation &annotation, std::size_t ordinal) {
    static const std::array<ImVec4, 8> palette{{
        ImVec4(1.00F, 0.70F, 0.15F, 0.92F),
        ImVec4(0.25F, 0.75F, 1.00F, 0.92F),
        ImVec4(0.95F, 0.40F, 0.65F, 0.92F),
        ImVec4(0.35F, 0.90F, 0.45F, 0.92F),
        ImVec4(0.80F, 0.55F, 1.00F, 0.92F),
        ImVec4(0.95F, 0.90F, 0.25F, 0.92F),
        ImVec4(0.20F, 0.85F, 0.85F, 0.92F),
        ImVec4(1.00F, 0.50F, 0.25F, 0.92F),
    }};

    auto palette_index = ordinal;
    if (annotation.target_index && *annotation.target_index >= 0) {
        palette_index = static_cast<std::size_t>(*annotation.target_index);
    }
    return palette[palette_index % palette.size()];
}

struct AnnotationStyle {
    std::string label;
    ImVec4 color;
    bool visible = true;
};

[[nodiscard]] std::vector<AnnotationStyle> annotation_styles_for(const TracePlotSnapshot &snapshot) {
    std::vector<AnnotationStyle> styles;
    styles.reserve(snapshot.annotations.size());

    for (const auto &annotation : snapshot.annotations) {
        const auto label = annotation_legend_label_for(annotation);
        const auto duplicate =
            std::find_if(styles.begin(), styles.end(), [&label](const auto &style) { return style.label == label; });
        if (label.empty() || duplicate != styles.end()) {
            continue;
        }

        styles.push_back(AnnotationStyle{
            .label = label,
            .color = annotation_marker_color(annotation, styles.size()),
        });
    }

    return styles;
}

[[nodiscard]] ImVec4 annotation_color_for(const std::vector<AnnotationStyle> &styles,
                                          const TracePlotAnnotation &annotation, std::size_t fallback_ordinal) {
    const auto label = annotation_legend_label_for(annotation);
    const auto style = std::find_if(styles.begin(), styles.end(),
                                    [&label](const auto &candidate) { return candidate.label == label; });
    if (style != styles.end()) {
        return style->color;
    }

    return annotation_marker_color(annotation, fallback_ordinal);
}

void plot_annotation_legend_items(std::vector<AnnotationStyle> &styles) {
    static constexpr std::array<double, 1> legend_dummy_point{0.0};

    for (auto &style : styles) {
        ImPlotSpec spec;
        spec.LineColor = style.color;
        spec.FillColor = style.color;
        spec.Flags = ImPlotItemFlags_NoFit;
        spec.Marker = ImPlotMarker_Circle;
        spec.MarkerFillColor = style.color;
        spec.MarkerLineColor = ImVec4(0.0F, 0.0F, 0.0F, 0.75F);
        ImPlot::PlotScatter(style.label.c_str(), legend_dummy_point.data(), legend_dummy_point.data(), 0, spec);
        if (const auto *item = ImPlot::GetItem(style.label.c_str())) {
            style.visible = item->Show;
        }
    }
}

[[nodiscard]] bool annotation_is_visible(const std::vector<AnnotationStyle> &styles,
                                         const TracePlotAnnotation &annotation) {
    const auto label = annotation_legend_label_for(annotation);
    const auto style = std::find_if(styles.begin(), styles.end(),
                                    [&label](const auto &candidate) { return candidate.label == label; });
    return style == styles.end() || style->visible;
}

[[nodiscard]] std::pair<double, double> trace_y_range(const float *values, std::int64_t count) {
    const auto *begin = values;
    const auto *end = values + count;
    const auto [minimum, maximum] = std::minmax_element(begin, end);
    return {*minimum, *maximum};
}

[[nodiscard]] std::optional<std::pair<double, double>>
centered_y_range_for(TraceView &view, const std::vector<const TracePlotSnapshot *> &snapshots) {
    auto lower = std::numeric_limits<double>::infinity();
    auto higher = -std::numeric_limits<double>::infinity();
    auto found = false;

    for (const auto *snapshot : snapshots) {
        if (!snapshot->center0) {
            continue;
        }
        const auto trace_index = selected_trace_index(view, *snapshot);
        const auto *values = snapshot->trace_data(trace_index);
        for (std::int64_t index = 0; index < snapshot->sample_count(); ++index) {
            const auto value = static_cast<double>(values[index]);
            if (!std::isfinite(value)) {
                continue;
            }
            lower = std::min(lower, value);
            higher = std::max(higher, value);
            found = true;
        }
    }

    if (!found) {
        return std::nullopt;
    }
    return trace_plot_centered_y_range(lower, higher);
}

[[nodiscard]] double normalized_annotation_y(const TracePlotAnnotation &annotation, std::pair<double, double> y_range) {
    const auto [minimum, maximum] = y_range;
    if (minimum == maximum) {
        return minimum;
    }
    return trace_plot_annotation_y_from_norm(*annotation.norm_value, minimum, maximum);
}

struct AnnotationCandidate {
    const TracePlotAnnotation *annotation = nullptr;
    ImVec2 anchor;
    double x_value = 0.0;
    std::size_t ordinal = 0;
};

[[nodiscard]] AnnotationCandidate annotation_candidate_for(const TracePlotSnapshot &snapshot,
                                                           const TracePlotAnnotation &annotation,
                                                           const std::vector<float> &x_values,
                                                           std::pair<double, double> y_range, float plot_top,
                                                           float marker_radius, std::size_t ordinal) {
    const auto x = annotation_x_value(snapshot, annotation.sample_index, x_values);
    if (annotation.value) {
        return AnnotationCandidate{
            .annotation = &annotation,
            .anchor = ImPlot::PlotToPixels(x, *annotation.value),
            .x_value = x,
            .ordinal = ordinal,
        };
    }
    if (annotation.norm_value) {
        return AnnotationCandidate{
            .annotation = &annotation,
            .anchor = ImPlot::PlotToPixels(x, normalized_annotation_y(annotation, y_range)),
            .x_value = x,
            .ordinal = ordinal,
        };
    }

    const auto limits = ImPlot::GetPlotLimits();
    auto anchor = ImPlot::PlotToPixels(x, limits.Y.Max);
    anchor.y = plot_top + marker_radius + 1.0F;
    return AnnotationCandidate{
        .annotation = &annotation,
        .anchor = anchor,
        .x_value = x,
        .ordinal = ordinal,
    };
}

[[nodiscard]] bool annotation_candidate_is_close(const AnnotationCandidate &left, const AnnotationCandidate &right) {
    static constexpr auto close_x_pixels = 7.0F;
    static constexpr auto close_y_pixels = 9.0F;
    return std::abs(left.anchor.x - right.anchor.x) <= close_x_pixels &&
           std::abs(left.anchor.y - right.anchor.y) <= close_y_pixels;
}

[[nodiscard]] bool annotation_candidates_share_marker(const AnnotationCandidate &left,
                                                      const AnnotationCandidate &right) {
    static constexpr auto same_marker_pixels = 0.5F;
    return std::abs(left.anchor.x - right.anchor.x) <= same_marker_pixels &&
           std::abs(left.anchor.y - right.anchor.y) <= same_marker_pixels;
}

[[nodiscard]] bool annotation_group_shares_marker(const std::vector<AnnotationCandidate> &group) {
    return std::all_of(group.begin(), group.end(), [&group](const auto &candidate) {
        return annotation_candidates_share_marker(candidate, group.front());
    });
}

[[nodiscard]] ImVec2 annotation_group_label_anchor(const std::vector<AnnotationCandidate> &group) {
    auto x = 0.0F;
    auto y = 0.0F;
    for (const auto &candidate : group) {
        x += candidate.anchor.x;
        y += candidate.anchor.y;
    }

    const auto count = static_cast<float>(group.size());
    return ImVec2(x / count, y / count);
}

[[nodiscard]] std::string annotation_group_tooltip_for(const TracePlotSnapshot &snapshot,
                                                       const std::vector<AnnotationCandidate> &group) {
    std::string tooltip;
    for (std::size_t index = 0; index < group.size(); ++index) {
        if (!tooltip.empty()) {
            tooltip += "\n\n";
        }
        tooltip += "[" + std::to_string(index + 1U) + "] ";
        tooltip += annotation_tooltip_for(snapshot, *group[index].annotation, group[index].x_value);
    }
    return tooltip;
}

void plot_snapshot_annotations(const TracePlotSnapshot &snapshot, const float *values,
                               const std::vector<float> &x_values,
                               const std::vector<AnnotationStyle> &annotation_styles,
                               std::int64_t selected_trace) {
    if (snapshot.annotations.empty()) {
        return;
    }

    struct HoveredAnnotation {
        float distance_squared = 0.0F;
        std::string tooltip;
    };

    auto *draw_list = ImPlot::GetPlotDrawList();
    const auto mouse = ImGui::GetMousePos();
    const auto plot_hovered = ImPlot::IsPlotHovered();
    std::optional<HoveredAnnotation> hovered;
    static constexpr auto marker_radius = 3.5F;
    static constexpr auto hover_radius = 8.0F;
    static constexpr auto hover_radius_squared = hover_radius * hover_radius;
    const auto plot_pos = ImPlot::GetPlotPos();
    const auto plot_top = plot_pos.y;
    const auto y_range = trace_y_range(values, snapshot.sample_count());

    std::vector<std::vector<AnnotationCandidate>> groups;
    for (std::size_t ordinal = 0; ordinal < snapshot.annotations.size(); ++ordinal) {
        const auto &annotation = snapshot.annotations[ordinal];
        // History: a row-bound annotation only shows on its own trace; -1 shows on all.
        if (annotation.trace_row >= 0 && annotation.trace_row != selected_trace) {
            continue;
        }
        if (!annotation_is_visible(annotation_styles, annotation)) {
            continue;
        }

        auto candidate =
            annotation_candidate_for(snapshot, annotation, x_values, y_range, plot_top, marker_radius, ordinal);
        auto grouped = false;
        for (auto &group : groups) {
            if (annotation_candidate_is_close(candidate, group.front())) {
                group.push_back(candidate);
                grouped = true;
                break;
            }
        }
        if (!grouped) {
            groups.push_back({candidate});
        }
    }

    ImPlot::PushPlotClipRect();
    for (const auto &group : groups) {
        if (group.size() > 1 && annotation_group_shares_marker(group)) {
            const auto marker = group.front().anchor;
            std::vector<ImVec4> colors;
            colors.reserve(group.size());
            for (const auto &candidate : group) {
                colors.push_back(annotation_color_for(annotation_styles, *candidate.annotation, candidate.ordinal));
            }
            draw_annotation_marker(*draw_list, marker, marker_radius, colors, TracePlotAnnotationMarker::Circle);
            draw_annotation_number_label(*draw_list, marker, std::to_string(group.size()));

            if (plot_hovered) {
                const auto distance = squared_distance(mouse, marker);
                if (distance <= hover_radius_squared && (!hovered || distance < hovered->distance_squared)) {
                    hovered = HoveredAnnotation{
                        .distance_squared = distance,
                        .tooltip = annotation_group_tooltip_for(snapshot, group),
                    };
                }
            }
            continue;
        }

        for (std::size_t index = 0; index < group.size(); ++index) {
            const auto &candidate = group[index];
            const auto &annotation = *candidate.annotation;
            const auto marker = candidate.anchor;
            const auto color = annotation_color_for(annotation_styles, annotation, candidate.ordinal);
            draw_annotation_marker(*draw_list, marker, marker_radius, {color}, annotation.marker);

            if (plot_hovered) {
                const auto distance = squared_distance(mouse, marker);
                if (distance <= hover_radius_squared && (!hovered || distance < hovered->distance_squared)) {
                    hovered = HoveredAnnotation{
                        .distance_squared = distance,
                        .tooltip = annotation_tooltip_for(snapshot, annotation, candidate.x_value),
                    };
                }
            }
        }
        if (group.size() > 1) {
            draw_annotation_number_label(*draw_list, annotation_group_label_anchor(group),
                                         std::to_string(group.size()));
        }
    }
    ImPlot::PopPlotClipRect();

    if (hovered) {
        ImGui::SetTooltip("%s", hovered->tooltip.c_str());
    }
}

// The ImPlot legend label/ID for a trace line (display name + a unique id suffix).
[[nodiscard]] std::string snapshot_plot_label(const TracePlotSnapshot &snapshot) {
    return snapshot_label(snapshot) + "##trace_plot_" + std::to_string(snapshot.id);
}

void plot_snapshot(TraceView &view, const TracePlotSnapshot &snapshot, const TracePlotSnapshot &axis_source) {
    const auto trace_index = selected_trace_index(view, snapshot);
    const auto *values = snapshot.trace_data(trace_index);
    const auto count = static_cast<int>(snapshot.sample_count());
    const auto label = snapshot_plot_label(snapshot);
    const auto &display_state = view.mutable_trace_display_state(snapshot);

    ImPlotSpec spec;
    spec.LineColor =
        ImVec4(display_state.color.red, display_state.color.green, display_state.color.blue, display_state.alpha);
    spec.LineWeight = static_cast<float>(snapshot.line_width);

    auto x_values = x_values_for(snapshot, axis_source);
    if (!x_values.empty()) {
        ImPlot::PlotLine(label.c_str(), x_values.data(), values, count, spec);
    } else {
        ImPlot::PlotLine(label.c_str(), values, count, 1.0, 0.0, spec);
    }

    auto annotation_styles = annotation_styles_for(snapshot);
    plot_annotation_legend_items(annotation_styles);
    plot_snapshot_annotations(snapshot, values, x_values, annotation_styles, trace_index);
}

[[nodiscard]] std::string panel_title(std::string_view group, const std::vector<const TracePlotSnapshot *> &snapshots,
                                      std::size_t panel_index) {
    const auto suffix = "##panel_" + std::to_string(panel_index) + "_" + std::string(group);
    if (snapshots.size() == 1) {
        return snapshot_label(*snapshots.front()) + suffix;
    }

    // Overlay sub-group: show the unique trace titles concatenated.
    std::vector<std::string> titles;
    for (const auto *snapshot : snapshots) {
        if (!snapshot->title.empty() && std::find(titles.begin(), titles.end(), snapshot->title) == titles.end()) {
            titles.push_back(snapshot->title);
        }
    }
    if (titles.empty()) {
        return group_title(group, snapshots) + suffix;
    }

    std::string joined;
    for (std::size_t index = 0; index < titles.size(); ++index) {
        if (index != 0) {
            joined += " | ";
        }
        joined += titles[index];
    }
    return joined + suffix;
}

// Right-click a legend entry for bulk visibility: isolate one, or show/hide all the
// panel's items (trace lines + annotation styles). Must be called after the items are
// plotted (SetupLock done) so GetItem resolves; ImPlot resolves items by
// ImGui::GetID(label), which differs inside the popup window, so the item pointers are
// cached here (the plot's ID context) and Show is set on them from the popup.
void draw_trace_legend_popups(const std::vector<const TracePlotSnapshot *> &snapshots) {
    std::vector<std::string> labels;
    for (const auto *snapshot : snapshots) {
        labels.push_back(snapshot_plot_label(*snapshot));
        for (const auto &style : annotation_styles_for(*snapshot)) {
            if (std::find(labels.begin(), labels.end(), style.label) == labels.end()) {
                labels.push_back(style.label);
            }
        }
    }
    std::vector<ImPlotItem *> items(labels.size(), nullptr);
    for (std::size_t index = 0; index < labels.size(); ++index) {
        items[index] = ImPlot::GetItem(labels[index].c_str());
    }
    const auto set_show = [&items](const auto &pred) {
        for (std::size_t index = 0; index < items.size(); ++index) {
            if (items[index] != nullptr) {
                items[index]->Show = pred(index);
            }
        }
    };
    for (std::size_t index = 0; index < labels.size(); ++index) {
        if (ImPlot::BeginLegendPopup(labels[index].c_str())) {
            if (ImGui::MenuItem("Isolate this")) {
                set_show([index](std::size_t other) { return other == index; });
            }
            if (ImGui::MenuItem("Show all")) {
                set_show([](std::size_t) { return true; });
            }
            if (ImGui::MenuItem("Hide all")) {
                set_show([](std::size_t) { return false; });
            }
            ImPlot::EndLegendPopup();
        }
    }
}

void draw_trace_plot_panel(TraceView &view, std::string_view group,
                           const std::vector<const TracePlotSnapshot *> &snapshots, std::size_t panel_index,
                           float plot_height) {
    const auto &front = *snapshots.front();
    const auto x_label = x_axis_label(front);
    const auto y_label = y_axis_label(front);
    const auto title = panel_title(group, snapshots, panel_index);

    const bool time_effective = front.x_axis == TracePlotXAxis::TimeUs && front.sample_rate_hz.has_value();
    const double sample_rate = front.sample_rate_hz.value_or(0.0);
    auto &axis_view = view.mutable_axis_view(front.id);

    if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1.0F, plot_height))) {
        // center0 snapshots force a symmetric Y range; otherwise the Y axis auto-fits
        // the data each frame. Without the AutoFit flag ImPlot keeps stale limits
        // (e.g. a previous center0 run), so center0=false would appear centered.
        const auto y_limits = centered_y_range_for(view, snapshots);
        const auto y_flags = y_limits ? ImPlotAxisFlags_None : ImPlotAxisFlags_AutoFit;
        ImPlot::SetupAxes(x_label.c_str(), y_label.c_str(), ImPlotAxisFlags_None, y_flags);
        if (y_limits) {
            ImPlot::SetupAxisLimits(ImAxis_Y1, y_limits->first, y_limits->second, ImPlotCond_Always);
        }

        // Keep the same visible window when the panel switches between sample and
        // time_us units: convert the stored X-limits by the sample<->time factor
        // so the data is not squished on the unit change.
        if (axis_view.initialized && axis_view.time_effective != time_effective && sample_rate > 0.0) {
            const double factor = time_effective ? (1.0e6 / sample_rate) : (sample_rate / 1.0e6);
            ImPlot::SetupAxisLimits(ImAxis_X1, axis_view.x_min * factor, axis_view.x_max * factor,
                                    ImPlotCond_Always);
        }

        for (const auto *snapshot : snapshots) {
            plot_snapshot(view, *snapshot, front);
        }
        draw_trace_legend_popups(snapshots);

        const auto limits = ImPlot::GetPlotLimits();
        axis_view.x_min = limits.X.Min;
        axis_view.x_max = limits.X.Max;
        axis_view.time_effective = time_effective;
        axis_view.initialized = true;

        ImPlot::EndPlot();
    }
}

void draw_trace_panel(TraceView &view, std::string_view group, const TracePanel &panel,
                      const std::vector<const TracePlotSnapshot *> &group_snapshots, std::size_t panel_index) {
    const auto plot_height = panel.snapshots.size() == 1 ? 260.0F : 340.0F;
    const auto controls = trace_control_count(panel.snapshots);
    if (controls == 0) {
        draw_trace_plot_panel(view, group, panel.snapshots, panel_index, plot_height);
        return;
    }

    const auto table_id = "trace_panel_" + std::to_string(panel_index) + "_" + std::string(group);
    if (ImGui::BeginTable(table_id.c_str(), 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("plot", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("trace", ImGuiTableColumnFlags_WidthFixed, std::max(54.0F, controls * 50.0F));
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        draw_trace_plot_panel(view, group, panel.snapshots, panel_index, plot_height);
        ImGui::TableSetColumnIndex(1);
        draw_vertical_trace_controls(view, group, panel.snapshots, group_snapshots, plot_height);
        ImGui::EndTable();
    }
}

[[nodiscard]] bool draw_gear_button(const char *id) {
    const auto size = ImGui::GetFrameHeight();
    const auto origin = ImGui::GetCursorScreenPos();
    const auto clicked = ImGui::InvisibleButton(id, ImVec2(size, size));
    const auto color = ImGui::GetColorU32(ImGui::IsItemHovered() ? ImGuiCol_ButtonHovered : ImGuiCol_Text);
    auto *draw_list = ImGui::GetWindowDrawList();
    const ImVec2 center(origin.x + size * 0.5F, origin.y + size * 0.5F);

    constexpr auto pi = 3.14159265358979323846F;
    const auto outer = size * 0.32F;
    const auto inner = size * 0.14F;
    for (int tooth = 0; tooth < 8; ++tooth) {
        const auto angle = static_cast<float>(tooth) * pi * 0.25F;
        const ImVec2 begin(center.x + std::cos(angle) * (outer - 1.0F), center.y + std::sin(angle) * (outer - 1.0F));
        const ImVec2 end(center.x + std::cos(angle) * (outer + 1.6F), center.y + std::sin(angle) * (outer + 1.6F));
        draw_list->AddLine(begin, end, color, 1.4F);
    }
    draw_list->AddCircle(center, outer - 0.5F, color, 16, 1.5F);
    draw_list->AddCircle(center, inner, color, 12, 1.4F);
    return clicked;
}

void draw_group_controls_window(TraceView &view, std::string_view group, const std::string &title,
                                const std::vector<const TracePlotSnapshot *> &snapshots,
                                PipelineControlRuntime *control_runtime) {
    auto &open = view.mutable_group_controls_open(group);
    if (!open) {
        return;
    }

    const auto window_title = "TracePlot controls - " + title + "##traceplot_controls_" + std::string(group);
    ImGui::SetNextWindowSize(ImVec2(420.0F, 320.0F), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(window_title.c_str(), &open)) {
        auto ordered = ordered_snapshots(view, snapshots);
        std::optional<std::pair<std::uint64_t, std::uint64_t>> requested_move;

        if (has_trace_sliders(ordered)) {
            auto &group_trace_lock = view.mutable_group_trace_lock(group, default_group_trace_lock(ordered));
            ImGui::Checkbox("Lock trace sliders", &group_trace_lock);
            ImGui::Separator();
        }

        ImGui::TextDisabled("Drag to reorder; click the gear to edit a TracePlot.");

        for (const auto *snapshot : ordered) {
            const auto &display_state = view.mutable_trace_display_state(*snapshot);
            const auto name = snapshot_display_name(*snapshot);

            ImGui::PushID(static_cast<int>(snapshot->id));

            // Gear -> open the element's full control window.
            const bool has_control =
                control_runtime != nullptr && control_runtime->has_element(snapshot->element_name);
            ImGui::BeginDisabled(!has_control);
            if (draw_gear_button("##gear") && has_control) {
                control_runtime->open(snapshot->element_name);
            }
            ImGui::EndDisabled();
            if (has_control && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Open %s controls", snapshot->element_name.c_str());
            }

            // Color swatch preview (shows the auto-assigned palette color too).
            ImGui::SameLine();
            const ImVec4 swatch(display_state.color.red, display_state.color.green, display_state.color.blue,
                                display_state.alpha);
            ImGui::ColorButton("##swatch", swatch, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker,
                               ImVec2(14.0F, 14.0F));

            // Name, draggable to reorder.
            ImGui::SameLine();
            ImGui::Selectable(name.c_str());
            if (ImGui::BeginDragDropSource()) {
                const auto dragged_id = snapshot->id;
                ImGui::SetDragDropPayload("LEAKFLOW_TRACEPLOT_ID", &dragged_id, sizeof(dragged_id));
                ImGui::TextUnformatted(name.c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginDragDropTarget()) {
                if (const auto *payload = ImGui::AcceptDragDropPayload("LEAKFLOW_TRACEPLOT_ID");
                    payload != nullptr && payload->DataSize == sizeof(std::uint64_t)) {
                    const auto dragged_id = *static_cast<const std::uint64_t *>(payload->Data);
                    requested_move = std::pair{dragged_id, snapshot->id};
                }
                ImGui::EndDragDropTarget();
            }

            ImGui::PopID();
        }

        if (requested_move) {
            move_snapshot_before(view, std::move(ordered), requested_move->first, requested_move->second);
        }
    }

    ImGui::End();
}

void draw_group(TraceView &view, std::string_view group, const std::vector<const TracePlotSnapshot *> &snapshots,
                int group_index, PipelineControlRuntime *control_runtime) {
    if (snapshots.empty()) {
        return;
    }

    auto ordered = ordered_snapshots(view, snapshots);
    const auto title = group_title(group, ordered);
    const auto window_id = title + "##traceplot_group_" + std::string(group);
    ImGui::SetNextWindowPos(ImVec2(24.0F + 36.0F * group_index, 24.0F + 36.0F * group_index), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(980.0F, 620.0F), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(window_id.c_str())) {
        if (ImGui::Button("Controls")) {
            view.mutable_group_controls_open(group) = true;
        }
        ImGui::SameLine();
        const auto group_text = "group: " + std::string(group);
        ImGui::TextUnformatted(group_text.c_str());

        const auto panels = build_trace_panels(ordered);
        for (std::size_t index = 0; index < panels.size(); ++index) {
            if (index != 0) {
                ImGui::Spacing();
            }
            draw_trace_panel(view, group, panels[index], ordered, index);
        }
    }
    ImGui::End();

    draw_group_controls_window(view, group, title, ordered, control_runtime);
}

} // namespace

std::uint64_t TraceView::add_trace(TracePlotSnapshot snapshot) {
    const auto lock = std::scoped_lock(mutex_);
    validate_snapshot(snapshot);
    if (snapshot.group.empty()) {
        snapshot.group = "default";
    }
    if (snapshot.label.empty()) {
        snapshot.label = "trace";
    }

    // accumulate (TracePlot update_mode=accumulate): grow this element's single
    // snapshot by appending the incoming buffer's rows as new traces, so the
    // [trace, sample] matrix builds a history the vertical slider can scrub.
    // Annotations are tagged with their trace row so only the selected trace's
    // markers show. The slider auto-follows the newest trace (see the renderer).
    if (snapshot.accumulate && !snapshot.element_name.empty()) {
        for (auto &existing : trace_snapshots_) {
            if (existing.element_name != snapshot.element_name) {
                continue;
            }
            if (existing.sample_count() != snapshot.sample_count()) {
                throw std::invalid_argument("TracePlot accumulate requires a constant sample count across buffers");
            }
            const auto id = existing.id;
            const auto base_row = existing.trace_count();
            const auto added_rows = snapshot.trace_count();
            const auto samples = existing.sample_count();
            existing.values.insert(existing.values.end(), snapshot.values.begin(), snapshot.values.end());
            existing.shape = {base_row + added_rows, samples};
            for (auto &annotation : snapshot.annotations) {
                annotation.trace_row = base_row + std::max<std::int64_t>(annotation.trace_row, 0);
                existing.annotations.push_back(std::move(annotation));
            }
            // The renderer owns slider follow (it rides the newest trace only while
            // streaming, and pauses on hold / in Idle), so no index update here.

            log::LogRecord append_record{
                .level = log::LogLevel::Info,
                .component = "plot",
                .message = "appended TracePlot trace",
                .fields =
                    {
                        {"group", existing.group},
                        {"label", existing.label},
                        {"traces", std::to_string(existing.trace_count())},
                        {"samples", std::to_string(samples)},
                        {"annotations", std::to_string(existing.annotations.size())},
                    },
            };
            log::write(std::move(append_record));
            return id;
        }
    }

    // Replay-safe update (Phase 25): if this element already registered a
    // snapshot, replace its data in place rather than appending a new trace.
    // Preserve the existing id and user-adjusted display state (color, alpha,
    // order, trace index) so a downstream rerun updates the curve/annotations
    // without duplicating it. See docs/design/pipeline_controller.md.
    if (!snapshot.accumulate && !snapshot.element_name.empty()) {
        for (auto &existing : trace_snapshots_) {
            if (existing.element_name != snapshot.element_name) {
                continue;
            }
            const auto id = existing.id;
            snapshot.id = id;
            const bool x_axis_changed = snapshot.x_axis != existing.x_axis;
            if (!snapshot.color) {
                snapshot.color = existing.color;
            }
            auto &lock = group_trace_locks_.try_emplace(snapshot.group, snapshot.lock_trace_index).first->second;
            lock = lock || snapshot.lock_trace_index;

            // Sync the live vertical slider only when the trace_index property
            // actually changed, so a data rerun preserves a manually-dragged
            // slider but a property edit moves it.
            if (snapshot.initial_trace_index != existing.initial_trace_index) {
                const auto trace_count = std::max<std::int64_t>(snapshot.trace_count(), 1);
                auto clamped = snapshot.initial_trace_index;
                clamped = std::max<std::int64_t>(clamped, 0);
                clamped = std::min<std::int64_t>(clamped, trace_count - 1);
                trace_indices_[id] = static_cast<int>(clamped);
            }

            // Likewise, sync the live line color/alpha only when an explicit color
            // property changed (a data rerun with an unchanged or auto color keeps
            // a color tweaked through the floating controls window).
            if (snapshot.color && existing.color && *snapshot.color != *existing.color) {
                auto &display_state = trace_display_states_[id];
                display_state.color = *snapshot.color;
                display_state.alpha = static_cast<float>(snapshot.alpha);
            }

            log::LogRecord update_record{
                .level = log::LogLevel::Info,
                .component = "plot",
                .message = "updated TracePlot snapshot",
                .fields =
                    {
                        {"group", snapshot.group},
                        {"label", snapshot.label},
                        {"traces", std::to_string(snapshot.trace_count())},
                        {"samples", std::to_string(snapshot.sample_count())},
                        {"annotations", std::to_string(snapshot.annotations.size())},
                    },
            };
            log::write(std::move(update_record));

            existing = std::move(snapshot);

            // An overlay sub-group shares one x-axis. When this element's x_axis
            // changed, push it to the other members of its overlay sub-group (the
            // panel) so they follow.
            if (x_axis_changed && x_axis_listener_) {
                std::vector<const TracePlotSnapshot *> group_members;
                for (const auto &candidate : trace_snapshots_) {
                    if (candidate.group == existing.group) {
                        group_members.push_back(&candidate);
                    }
                }
                const auto order_of = [this](const TracePlotSnapshot *snapshot_ptr) {
                    const auto found = trace_display_states_.find(snapshot_ptr->id);
                    return found != trace_display_states_.end() ? found->second.order : snapshot_ptr->order;
                };
                std::stable_sort(group_members.begin(), group_members.end(),
                                 [&order_of](const auto *left, const auto *right) {
                                     if (order_of(left) != order_of(right)) {
                                         return order_of(left) < order_of(right);
                                     }
                                     return left->id < right->id;
                                 });

                std::vector<std::vector<const TracePlotSnapshot *>> panels;
                for (std::size_t index = 0; index < group_members.size(); ++index) {
                    if (panels.empty() || (index != 0 && group_members[index]->layout == TracePlotLayout::Stacked)) {
                        panels.push_back({});
                    }
                    panels.back().push_back(group_members[index]);
                }

                for (const auto &panel : panels) {
                    const bool contains =
                        std::any_of(panel.begin(), panel.end(), [id](const auto *s) { return s->id == id; });
                    if (!contains) {
                        continue;
                    }
                    for (const auto *peer : panel) {
                        if (peer->id != id && peer->x_axis != existing.x_axis && !peer->element_name.empty()) {
                            x_axis_listener_(peer->element_name, to_string(existing.x_axis));
                        }
                    }
                    break;
                }
            }

            return id;
        }
    }

    snapshot.id = next_snapshot_id_++;
    if (!snapshot.color) {
        snapshot.color = default_trace_color(snapshot.id);
    }
    const auto id = snapshot.id;
    trace_indices_.emplace(id, static_cast<int>(snapshot.initial_trace_index));
    auto &group_trace_lock = group_trace_locks_.try_emplace(snapshot.group, snapshot.lock_trace_index).first->second;
    group_trace_lock = group_trace_lock || snapshot.lock_trace_index;
    trace_display_states_.emplace(id, TracePlotDisplayState{
                                          .color = *snapshot.color,
                                          .alpha = static_cast<float>(snapshot.alpha),
                                          .order = snapshot.order,
                                      });

    log::LogRecord record{
        .level = log::LogLevel::Info,
        .component = "plot",
        .message = "registered TracePlot snapshot",
        .fields =
            {
                {"group", snapshot.group},
                {"label", snapshot.label},
                {"rank", std::to_string(snapshot.rank())},
                {"traces", std::to_string(snapshot.trace_count())},
                {"samples", std::to_string(snapshot.sample_count())},
                {"layout", std::string(to_string(snapshot.layout))},
                {"order", std::to_string(snapshot.order)},
            },
    };
    log::write(std::move(record));

    // First accumulate buffer: bind its annotations to their trace rows so later
    // appended traces do not inherit them (-1 would draw on every trace).
    if (snapshot.accumulate) {
        for (auto &annotation : snapshot.annotations) {
            annotation.trace_row = std::max<std::int64_t>(annotation.trace_row, 0);
        }
    }

    trace_snapshots_.push_back(std::move(snapshot));
    return id;
}
bool TraceView::refresh_trace_display(const TracePlotSnapshot &update) {
    const auto lock = std::scoped_lock(mutex_);
    for (auto &existing : trace_snapshots_) {
        if (existing.element_name != update.element_name) {
            continue;
        }
        existing.group = update.group.empty() ? "default" : update.group;
        existing.label = update.label;
        existing.title = update.title;
        existing.x_label = update.x_label;
        existing.y_label = update.y_label;
        existing.layout = update.layout;
        existing.line_width = update.line_width;
        existing.order = update.order;
        existing.center0 = update.center0;
        existing.x_axis = update.x_axis;
        // sample rate: keep the captured (metadata) value when no property override.
        if (update.sample_rate_hz) {
            existing.sample_rate_hz = update.sample_rate_hz;
        }
        // color: keep the existing (auto palette or user-tweaked) color when the
        // color property is unset; otherwise apply the explicit color/alpha.
        if (update.color) {
            existing.color = update.color;
            auto &display_state = trace_display_states_[existing.id];
            display_state.color = *update.color;
            display_state.alpha = static_cast<float>(update.alpha);
        }
        // trace index: replace-mode follows the property; accumulate auto-follows so
        // it is left to the renderer.
        if (!existing.accumulate) {
            const auto trace_count = std::max<std::int64_t>(existing.trace_count(), 1);
            const auto clamped = std::clamp<std::int64_t>(update.initial_trace_index, 0, trace_count - 1);
            existing.initial_trace_index = clamped;
            trace_indices_[existing.id] = static_cast<int>(clamped);
        }
        // time_us needs a rate; without one fall back to sample and report it so the
        // caller resets the x_axis property (keeping control/graph/plot consistent).
        if (existing.x_axis == TracePlotXAxis::TimeUs && !existing.sample_rate_hz) {
            existing.x_axis = TracePlotXAxis::Sample;
            return true;
        }
        return false;
    }
    return false;
}
const std::vector<TracePlotSnapshot> &TraceView::trace_snapshots() const {
    const auto lock = std::scoped_lock(mutex_);
    return trace_snapshots_;
}
int &TraceView::mutable_trace_index(std::uint64_t snapshot_id, int initial_value) {
    const auto lock = std::scoped_lock(mutex_);
    return trace_indices_.try_emplace(snapshot_id, initial_value).first->second;
}
bool &TraceView::mutable_trace_follow_latest(std::uint64_t snapshot_id, bool initial_value) {
    const auto lock = std::scoped_lock(mutex_);
    return trace_follow_latest_.try_emplace(snapshot_id, initial_value).first->second;
}
bool TraceView::streaming() const {
    const auto lock = std::scoped_lock(mutex_);
    return streaming_;
}
bool &TraceView::mutable_group_trace_lock(std::string_view group, bool initial_value) {
    const auto lock = std::scoped_lock(mutex_);
    return group_trace_locks_.try_emplace(std::string(group), initial_value).first->second;
}
TracePlotDisplayState &TraceView::mutable_trace_display_state(const TracePlotSnapshot &snapshot) {
    const auto lock = std::scoped_lock(mutex_);
    return trace_display_states_
        .try_emplace(snapshot.id,
                     TracePlotDisplayState{
                         .color = snapshot.color.value_or(default_trace_color(snapshot.id)),
                         .alpha = static_cast<float>(snapshot.alpha),
                         .order = snapshot.order,
                     })
        .first->second;
}
bool &TraceView::mutable_group_controls_open(std::string_view group) {
    const auto lock = std::scoped_lock(mutex_);
    return group_control_windows_.try_emplace(std::string(group), false).first->second;
}
PlotAxisView &TraceView::mutable_axis_view(std::uint64_t panel_id) {
    const auto lock = std::scoped_lock(mutex_);
    return axis_views_[panel_id];
}
void TraceView::set_trace_index_listener(std::function<void(std::string_view, int)> listener) {
    const auto lock = std::scoped_lock(mutex_);
    trace_index_listener_ = std::move(listener);
}
void TraceView::set_x_axis_listener(std::function<void(std::string_view, std::string_view)> listener) {
    const auto lock = std::scoped_lock(mutex_);
    x_axis_listener_ = std::move(listener);
}
void TraceView::notify_x_axis(std::string_view element_name, std::string_view x_axis) {
    std::function<void(std::string_view, std::string_view)> listener;
    {
        const auto lock = std::scoped_lock(mutex_);
        listener = x_axis_listener_;
    }
    if (listener && !element_name.empty()) {
        listener(element_name, x_axis);
    }
}
void TraceView::notify_trace_index(std::string_view element_name, int trace_index) {
    std::function<void(std::string_view, int)> listener;
    {
        const auto lock = std::scoped_lock(mutex_);
        listener = trace_index_listener_;
    }
    if (listener && !element_name.empty()) {
        listener(element_name, trace_index);
    }
}

void TraceView::draw(const PlotDrawContext &context) {
    // UI thread. Lock our own data against the worker's add_trace/refresh; record the
    // streaming flag for accumulate sliders, then draw one window per group.
    const auto lock = std::scoped_lock(mutex_);
    streaming_ = context.streaming;
    if (trace_snapshots_.empty()) {
        return;
    }
    std::map<std::string, std::vector<const TracePlotSnapshot *>> groups;
    for (const auto &snapshot : trace_snapshots_) {
        groups[snapshot.group].push_back(&snapshot);
    }
    auto group_index = 0;
    for (const auto &[group, snapshots] : groups) {
        draw_group(*this, group, snapshots, group_index, context.control_runtime);
        ++group_index;
    }
}

void TraceView::clear() {
    const auto lock = std::scoped_lock(mutex_);
    trace_snapshots_.clear();
    // Reset the id counter so a re-registered plot reuses ids after a Stop/Start
    // cycle; otherwise auto palette colors (keyed by id) would shift each cycle.
    // Every id-keyed map is cleared so reused ids carry no stale state.
    next_snapshot_id_ = 1;
    trace_indices_.clear();
    trace_follow_latest_.clear();
    group_trace_locks_.clear();
    trace_display_states_.clear();
    group_control_windows_.clear();
    axis_views_.clear();
}

bool TraceView::empty() const {
    const auto lock = std::scoped_lock(mutex_);
    return trace_snapshots_.empty();
}

} // namespace leakflow::plot
