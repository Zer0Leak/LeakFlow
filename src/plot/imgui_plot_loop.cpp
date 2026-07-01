#include "leakflow/plot/plot_runtime.hpp"

#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/log/logger.hpp"
#include "leakflow/plot/pipeline_graph.hpp"
#include "leakflow/plot/plot_view.hpp"
#include "plot_render_util.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <implot_internal.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plot {
namespace {

#if defined(__linux__)
[[nodiscard]] bool environment_variable_is_set(const char *name) {
    const auto *value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

void require_display_environment() {
    if (environment_variable_is_set("DISPLAY")) {
        return;
    }
    if (environment_variable_is_set("WAYLAND_DISPLAY") && environment_variable_is_set("XDG_RUNTIME_DIR")) {
        return;
    }

    throw std::runtime_error("could not initialize GLFW for LeakFlow plot runtime: no DISPLAY or "
                             "WAYLAND_DISPLAY/XDG_RUNTIME_DIR is set. In Docker, start the container "
                             "with X11 forwarding (-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix) or "
                             "Wayland socket forwarding.");
}

void configure_glfw_platform_hint() {
#if defined(GLFW_PLATFORM)
    if (environment_variable_is_set("DISPLAY")) {
#if defined(GLFW_PLATFORM_X11)
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
        return;
    }

    if (environment_variable_is_set("WAYLAND_DISPLAY")) {
#if defined(GLFW_PLATFORM_WAYLAND)
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
#endif
    }
#endif
}
#endif

struct GlfwContext {
    GlfwContext() {
        glfwSetErrorCallback([](int, const char *description) {
            log::LogRecord record{
                .level = log::LogLevel::Error,
                .component = "plot",
                .message = "GLFW error",
                .fields =
                    {
                        {"description", description == nullptr ? "" : description},
                    },
            };
            log::write(std::move(record));
        });

#if defined(__linux__)
        require_display_environment();
        configure_glfw_platform_hint();
#endif

        if (glfwInit() == 0) {
            throw std::runtime_error("could not initialize GLFW for LeakFlow plot runtime; check display "
                                     "forwarding, X11/Wayland authorization, and Docker NVIDIA graphics "
                                     "driver capabilities");
        }
    }

    ~GlfwContext() { glfwTerminate(); }
};

struct GlfwWindow {
    GLFWwindow *window = nullptr;

    GlfwWindow(int width, int height, const std::string &title) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            throw std::runtime_error("could not create GLFW window for LeakFlow plot runtime");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
    }

    ~GlfwWindow() {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }
};

struct ImGuiContextOwner {
    explicit ImGuiContextOwner(GLFWwindow *window) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");
    }

    ~ImGuiContextOwner() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
};

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
ordered_snapshots(PlotRuntime &runtime, const std::vector<const TracePlotSnapshot *> &snapshots) {
    auto ordered = snapshots;
    std::stable_sort(ordered.begin(), ordered.end(), [&runtime](const auto *left, const auto *right) {
        const auto &left_state = runtime.mutable_trace_display_state(*left);
        const auto &right_state = runtime.mutable_trace_display_state(*right);
        if (left_state.order != right_state.order) {
            return left_state.order < right_state.order;
        }

        return left->id < right->id;
    });

    return ordered;
}

void apply_display_order(PlotRuntime &runtime, const std::vector<const TracePlotSnapshot *> &snapshots) {
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
        runtime.mutable_trace_display_state(*snapshots[index]).order = static_cast<std::int64_t>(index);
    }
}

void move_snapshot_before(PlotRuntime &runtime, std::vector<const TracePlotSnapshot *> snapshots,
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
    apply_display_order(runtime, snapshots);
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

[[nodiscard]] int selected_trace_index(PlotRuntime &runtime, const TracePlotSnapshot &snapshot) {
    if (snapshot.rank() == 1) {
        return 0;
    }

    auto &trace_index = runtime.mutable_trace_index(snapshot.id, static_cast<int>(snapshot.initial_trace_index));
    // An accumulate snapshot streams new traces in; while streaming, the slider rides
    // the newest one by default. The slider widget clears follow_latest while the user
    // holds it; once a run is held (Idle) or frozen (Paused) we stop following so the
    // history can be scrubbed freely.
    if (snapshot.accumulate && runtime.streaming() && runtime.mutable_trace_follow_latest(snapshot.id, true)) {
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

[[nodiscard]] int clamped_group_trace_delta(PlotRuntime &runtime,
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

        auto &trace_index = runtime.mutable_trace_index(snapshot->id, static_cast<int>(snapshot->initial_trace_index));
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

void apply_group_trace_delta(PlotRuntime &runtime,
                             const std::vector<const TracePlotSnapshot *> &snapshots,
                             int delta) {
    if (delta == 0) {
        return;
    }

    for (const auto *snapshot : snapshots) {
        if (!snapshot_has_trace_slider(*snapshot)) {
            continue;
        }

        auto &trace_index = runtime.mutable_trace_index(snapshot->id, static_cast<int>(snapshot->initial_trace_index));
        trace_index = clamped_trace_index(*snapshot, trace_index + delta);
    }
}

void draw_vertical_trace_controls(PlotRuntime &runtime, std::string_view group,
                                  const std::vector<const TracePlotSnapshot *> &snapshots,
                                  const std::vector<const TracePlotSnapshot *> &group_snapshots,
                                  float plot_height) {
    const auto slider_height = std::max(120.0F, plot_height);
    const auto group_has_multiple_sliders = trace_control_count(group_snapshots) > 1;
    auto drew_control = false;
    auto &group_trace_lock = runtime.mutable_group_trace_lock(group, default_group_trace_lock(group_snapshots));

    for (const auto *snapshot : snapshots) {
        if (!snapshot_has_trace_slider(*snapshot)) {
            continue;
        }

        if (drew_control) {
            ImGui::SameLine();
        }

        auto &trace_index = runtime.mutable_trace_index(snapshot->id, static_cast<int>(snapshot->initial_trace_index));
        trace_index = clamped_trace_index(*snapshot, trace_index);
        auto requested_trace_index = trace_index;
        const auto &display_state = runtime.mutable_trace_display_state(*snapshot);
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
                const auto delta = clamped_group_trace_delta(runtime, group_snapshots, requested_trace_index - trace_index);
                apply_group_trace_delta(runtime, group_snapshots, delta);
                // Push each affected slider back to its element's trace_index property.
                // Accumulate sliders scrub runtime history only, so they are not
                // written back (it would push an out-of-range index onto the next
                // single-buffer snapshot).
                for (const auto *grouped : group_snapshots) {
                    if (grouped->accumulate) {
                        continue;
                    }
                    runtime.notify_trace_index(
                        grouped->element_name,
                        runtime.mutable_trace_index(grouped->id, static_cast<int>(grouped->initial_trace_index)));
                }
            } else {
                trace_index = requested_trace_index;
                if (!snapshot->accumulate) {
                    runtime.notify_trace_index(snapshot->element_name, trace_index);
                }
            }
        }

        // Accumulate plots ride the newest trace; holding the slider (or a group
        // trace lock owning the index) pauses the follow so a past trace can be
        // inspected. On release it resumes and snaps back to the latest.
        if (snapshot->accumulate) {
            runtime.mutable_trace_follow_latest(snapshot->id, true) = !group_trace_lock && !slider_active;
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
centered_y_range_for(PlotRuntime &runtime, const std::vector<const TracePlotSnapshot *> &snapshots) {
    auto lower = std::numeric_limits<double>::infinity();
    auto higher = -std::numeric_limits<double>::infinity();
    auto found = false;

    for (const auto *snapshot : snapshots) {
        if (!snapshot->center0) {
            continue;
        }
        const auto trace_index = selected_trace_index(runtime, *snapshot);
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

void plot_snapshot(PlotRuntime &runtime, const TracePlotSnapshot &snapshot, const TracePlotSnapshot &axis_source) {
    const auto trace_index = selected_trace_index(runtime, snapshot);
    const auto *values = snapshot.trace_data(trace_index);
    const auto count = static_cast<int>(snapshot.sample_count());
    const auto label = snapshot_label(snapshot) + "##trace_plot_" + std::to_string(snapshot.id);
    const auto &display_state = runtime.mutable_trace_display_state(snapshot);

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

void draw_trace_plot_panel(PlotRuntime &runtime, std::string_view group,
                           const std::vector<const TracePlotSnapshot *> &snapshots, std::size_t panel_index,
                           float plot_height) {
    const auto &front = *snapshots.front();
    const auto x_label = x_axis_label(front);
    const auto y_label = y_axis_label(front);
    const auto title = panel_title(group, snapshots, panel_index);

    const bool time_effective = front.x_axis == TracePlotXAxis::TimeUs && front.sample_rate_hz.has_value();
    const double sample_rate = front.sample_rate_hz.value_or(0.0);
    auto &axis_view = runtime.mutable_axis_view(front.id);

    if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1.0F, plot_height))) {
        // center0 snapshots force a symmetric Y range; otherwise the Y axis auto-fits
        // the data each frame. Without the AutoFit flag ImPlot keeps stale limits
        // (e.g. a previous center0 run), so center0=false would appear centered.
        const auto y_limits = centered_y_range_for(runtime, snapshots);
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
            plot_snapshot(runtime, *snapshot, front);
        }

        const auto limits = ImPlot::GetPlotLimits();
        axis_view.x_min = limits.X.Min;
        axis_view.x_max = limits.X.Max;
        axis_view.time_effective = time_effective;
        axis_view.initialized = true;

        ImPlot::EndPlot();
    }
}

void draw_trace_panel(PlotRuntime &runtime, std::string_view group, const TracePanel &panel,
                      const std::vector<const TracePlotSnapshot *> &group_snapshots, std::size_t panel_index) {
    const auto plot_height = panel.snapshots.size() == 1 ? 260.0F : 340.0F;
    const auto controls = trace_control_count(panel.snapshots);
    if (controls == 0) {
        draw_trace_plot_panel(runtime, group, panel.snapshots, panel_index, plot_height);
        return;
    }

    const auto table_id = "trace_panel_" + std::to_string(panel_index) + "_" + std::string(group);
    if (ImGui::BeginTable(table_id.c_str(), 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("plot", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("trace", ImGuiTableColumnFlags_WidthFixed, std::max(54.0F, controls * 50.0F));
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        draw_trace_plot_panel(runtime, group, panel.snapshots, panel_index, plot_height);
        ImGui::TableSetColumnIndex(1);
        draw_vertical_trace_controls(runtime, group, panel.snapshots, group_snapshots, plot_height);
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

void draw_group_controls_window(PlotRuntime &runtime, std::string_view group, const std::string &title,
                                const std::vector<const TracePlotSnapshot *> &snapshots,
                                PipelineControlRuntime *control_runtime) {
    auto &open = runtime.mutable_group_controls_open(group);
    if (!open) {
        return;
    }

    const auto window_title = "TracePlot controls - " + title + "##traceplot_controls_" + std::string(group);
    ImGui::SetNextWindowSize(ImVec2(420.0F, 320.0F), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(window_title.c_str(), &open)) {
        auto ordered = ordered_snapshots(runtime, snapshots);
        std::optional<std::pair<std::uint64_t, std::uint64_t>> requested_move;

        if (has_trace_sliders(ordered)) {
            auto &group_trace_lock = runtime.mutable_group_trace_lock(group, default_group_trace_lock(ordered));
            ImGui::Checkbox("Lock trace sliders", &group_trace_lock);
            ImGui::Separator();
        }

        ImGui::TextDisabled("Drag to reorder; click the gear to edit a TracePlot.");

        for (const auto *snapshot : ordered) {
            const auto &display_state = runtime.mutable_trace_display_state(*snapshot);
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
            move_snapshot_before(runtime, std::move(ordered), requested_move->first, requested_move->second);
        }
    }

    ImGui::End();
}

void draw_group(PlotRuntime &runtime, std::string_view group, const std::vector<const TracePlotSnapshot *> &snapshots,
                int group_index, PipelineControlRuntime *control_runtime) {
    if (snapshots.empty()) {
        return;
    }

    auto ordered = ordered_snapshots(runtime, snapshots);
    const auto title = group_title(group, ordered);
    const auto window_id = title + "##traceplot_group_" + std::string(group);
    ImGui::SetNextWindowPos(ImVec2(24.0F + 36.0F * group_index, 24.0F + 36.0F * group_index), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(980.0F, 620.0F), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(window_id.c_str())) {
        if (ImGui::Button("Controls")) {
            runtime.mutable_group_controls_open(group) = true;
        }
        ImGui::SameLine();
        const auto group_text = "group: " + std::string(group);
        ImGui::TextUnformatted(group_text.c_str());

        const auto panels = build_trace_panels(ordered);
        for (std::size_t index = 0; index < panels.size(); ++index) {
            if (index != 0) {
                ImGui::Spacing();
            }
            draw_trace_panel(runtime, group, panels[index], ordered, index);
        }
    }
    ImGui::End();

    draw_group_controls_window(runtime, group, title, ordered, control_runtime);
}

void append_u32_be(std::vector<unsigned char> &output, std::uint32_t value) {
    output.push_back(static_cast<unsigned char>((value >> 24U) & 0xFFU));
    output.push_back(static_cast<unsigned char>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<unsigned char>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<unsigned char>(value & 0xFFU));
}

void append_chunk(std::vector<unsigned char> &png, const char *type, const std::vector<unsigned char> &data) {
    append_u32_be(png, static_cast<std::uint32_t>(data.size()));

    const auto type_begin = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());

    const auto crc =
        crc32(0, reinterpret_cast<const Bytef *>(png.data() + type_begin), static_cast<uInt>(png.size() - type_begin));
    append_u32_be(png, static_cast<std::uint32_t>(crc));
}

void save_rgba_png(const std::filesystem::path &path, int width, int height,
                   const std::vector<unsigned char> &rgba_bottom_up) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("PNG dimensions must be positive");
    }

    const auto stride = static_cast<std::size_t>(width) * 4U;
    std::vector<unsigned char> scanlines((stride + 1U) * static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        const auto destination = static_cast<std::size_t>(row) * (stride + 1U);
        const auto source = static_cast<std::size_t>(height - row - 1) * stride;
        scanlines[destination] = 0;
        std::memcpy(scanlines.data() + destination + 1U, rgba_bottom_up.data() + source, stride);
    }

    auto compressed_size = compressBound(static_cast<uLong>(scanlines.size()));
    std::vector<unsigned char> compressed(compressed_size);
    const auto compressed_result = compress2(compressed.data(), &compressed_size, scanlines.data(),
                                             static_cast<uLong>(scanlines.size()), Z_BEST_SPEED);
    if (compressed_result != Z_OK) {
        throw std::runtime_error("could not compress PNG image data");
    }
    compressed.resize(compressed_size);

    std::vector<unsigned char> png{
        0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n',
    };

    std::vector<unsigned char> ihdr;
    append_u32_be(ihdr, static_cast<std::uint32_t>(width));
    append_u32_be(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8);
    ihdr.push_back(6);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);

    append_chunk(png, "IHDR", ihdr);
    append_chunk(png, "IDAT", compressed);
    append_chunk(png, "IEND", {});

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not open PNG output path");
    }
    output.write(reinterpret_cast<const char *>(png.data()), static_cast<std::streamsize>(png.size()));
    if (!output) {
        throw std::runtime_error("could not write PNG output path");
    }
}

void save_current_framebuffer_png(const std::filesystem::path &path, int width, int height) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    save_rgba_png(path, width, height, pixels);
}

} // namespace

void draw_plot_runtime(PlotRuntime &runtime, PipelineControlRuntime *control_runtime) {
    const auto lock = std::scoped_lock(runtime.mutex());
    // Accumulate sliders auto-follow the newest trace only while the session is
    // Running; in Idle/Paused (or a plain non-session plot run) they stay put.
    runtime.set_streaming(control_runtime != nullptr && control_runtime->session() != nullptr &&
                          control_runtime->session()->state() == leakflow::PipelineSessionState::Running);
    if (!runtime.has_sessions()) {
        ImGui::Begin("LeakFlow TracePlot");
        ImGui::TextUnformatted("No plot sessions");
        ImGui::End();
        return;
    }

    std::map<std::string, std::vector<const TracePlotSnapshot *>> groups;
    for (const auto &snapshot : runtime.trace_snapshots()) {
        groups[snapshot.group].push_back(&snapshot);
    }

    auto group_index = 0;
    for (const auto &[group, snapshots] : groups) {
        draw_group(runtime, group, snapshots, group_index, control_runtime);
        ++group_index;
    }

    // Registered views (e.g. ScoreView) draw themselves; the loop is type-agnostic.
    for (const auto &view : runtime.views()) {
        if (view) {
            view->draw();
        }
    }
}

void run_until_closed_impl(PlotRuntime &runtime, PipelineGraphRuntime *graph_runtime,
                           PipelineControlRuntime *control_runtime, const PlotLoopOptions &options) {
    const auto backend = options.backend == PlotBackend::Auto ? PlotBackend::OpenGL3 : options.backend;
    if (backend == PlotBackend::Vulkan) {
        throw std::runtime_error("the Vulkan plot backend is not built in Phase 22");
    }

    std::size_t initial_session_count = 0;
    {
        const auto lock = std::scoped_lock(runtime.mutex());
        initial_session_count = runtime.trace_snapshots().size();
    }

    log::LogRecord start_record{
        .level = log::LogLevel::Info,
        .component = "plot",
        .message = "starting plot runtime",
        .fields =
            {
                {"backend", std::string(to_string(backend))},
                {"sessions", std::to_string(initial_session_count)},
            },
    };
    log::write(std::move(start_record));

    GlfwContext glfw_context;
    GlfwWindow window(options.width, options.height, options.window_title);
    ImGuiContextOwner imgui_context(window.window);

    std::array<char, 512> png_path{};
    const auto default_path = std::string("leakflow_traceplot.png");
    std::copy(default_path.begin(), default_path.end(), png_path.begin());

    bool screenshot_requested = false;
    std::string screenshot_status;

    while (glfwWindowShouldClose(window.window) == 0) {
        if (options.should_close && options.should_close()) {
            glfwSetWindowShouldClose(window.window, GLFW_TRUE);
            break;
        }
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (options.show_controls_window) {
            const auto toolbar_title = options.window_title + std::string(" Controls");
            ImGui::Begin(toolbar_title.c_str());
            ImGui::InputText("PNG path", png_path.data(), png_path.size());
            ImGui::SameLine();
            if (ImGui::Button("Save PNG")) {
                screenshot_requested = true;
            }
            if (!screenshot_status.empty()) {
                ImGui::TextUnformatted(screenshot_status.c_str());
            }
            // Player controls (Stopped/Running/Paused/Idle). The UI calls the request;
            // the worker drives the state machine. Buttons enable/disable by state.
            if (control_runtime != nullptr && control_runtime->session() != nullptr) {
                ImGui::Separator();
                const auto state = control_runtime->session()->state();
                const bool stopped = state == leakflow::PipelineSessionState::Stopped;
                const bool running = state == leakflow::PipelineSessionState::Running;
                const bool paused = state == leakflow::PipelineSessionState::Paused;
                const bool idle = state == leakflow::PipelineSessionState::Idle;
                const char *label = stopped ? "Stopped" : running ? "Running" : paused ? "Paused" : "Idle";

                const auto gated_button = [](const char *text, bool enabled) {
                    ImGui::BeginDisabled(!enabled);
                    const bool clicked = ImGui::Button(text);
                    ImGui::EndDisabled();
                    return clicked;
                };

                ImGui::TextUnformatted(label);
                ImGui::SameLine();
                if (gated_button("Start", stopped)) { // Idle must Stop first; Stop recycles to a fresh run
                    control_runtime->request_start();
                }
                ImGui::SameLine();
                if (gated_button("Stop", running || paused || idle)) {
                    control_runtime->request_stop();
                }
                ImGui::SameLine();
                if (gated_button("Pause", running)) {
                    control_runtime->request_pause();
                }
                ImGui::SameLine();
                if (gated_button("Resume", paused)) {
                    control_runtime->request_resume();
                }

                // Auto-apply vs manual is orthogonal to the state: in manual mode edits
                // stage until Apply, in any state (Stopped/Running/Paused/Idle).
                bool auto_recompute = control_runtime->auto_recompute();
                if (ImGui::Checkbox("Auto-apply edits", &auto_recompute)) {
                    control_runtime->set_auto_recompute(auto_recompute);
                }
                if (!auto_recompute) {
                    ImGui::SameLine();
                    if (ImGui::Button("Apply")) {
                        control_runtime->request_apply();
                    }
                }
            }
            ImGui::End();
        }
        if (graph_runtime != nullptr) {
            draw_pipeline_graph(*graph_runtime, control_runtime);
        } else if (control_runtime != nullptr) {
            draw_pipeline_controls(*control_runtime);
        }
        if (graph_runtime == nullptr || runtime.has_sessions()) {
            draw_plot_runtime(runtime, control_runtime);
        }

        ImGui::Render();

        int display_width = 0;
        int display_height = 0;
        glfwGetFramebufferSize(window.window, &display_width, &display_height);
        glViewport(0, 0, display_width, display_height);
        glClearColor(0.09F, 0.10F, 0.11F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (screenshot_requested) {
            try {
                save_current_framebuffer_png(png_path.data(), display_width, display_height);
                screenshot_status = std::string("saved ") + png_path.data();
                log::LogRecord record{
                    .level = log::LogLevel::Info,
                    .component = "plot",
                    .message = "saved plot PNG",
                    .fields =
                        {
                            {"path", png_path.data()},
                        },
                };
                log::write(std::move(record));
            } catch (const std::exception &error) {
                screenshot_status = std::string("save failed: ") + error.what();
                log::LogRecord record{
                    .level = log::LogLevel::Error,
                    .component = "plot",
                    .message = "failed to save plot PNG",
                    .fields =
                        {
                            {"error", error.what()},
                        },
                };
                log::write(std::move(record));
            }
            screenshot_requested = false;
        }

        glfwSwapBuffers(window.window);
    }

    log::LogRecord finish_record{
        .level = log::LogLevel::Info,
        .component = "plot",
        .message = "stopped plot runtime",
    };
    log::write(std::move(finish_record));
}

void run_until_closed(PlotRuntime &runtime, PipelineGraphRuntime &graph_runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, &graph_runtime, nullptr, options);
}

void run_until_closed(PlotRuntime &runtime, PipelineGraphRuntime &graph_runtime,
                      PipelineControlRuntime &control_runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, &graph_runtime, &control_runtime, options);
}

void run_until_closed(PlotRuntime &runtime, PipelineControlRuntime &control_runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, nullptr, &control_runtime, options);
}

void run_until_closed(PlotRuntime &runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, nullptr, nullptr, options);
}

} // namespace leakflow::plot
