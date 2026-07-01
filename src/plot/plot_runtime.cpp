#include "leakflow/plot/plot_runtime.hpp"

#include "leakflow/log/logger.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow::plot {
namespace {

[[nodiscard]] std::string lower_string(std::string_view text) {
    std::string lowered;
    lowered.reserve(text.size());
    for (const auto character : text) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    return lowered;
}

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

} // namespace

std::int64_t TracePlotSnapshot::rank() const { return static_cast<std::int64_t>(shape.size()); }

std::int64_t TracePlotSnapshot::trace_count() const {
    if (shape.size() == 1) {
        return 1;
    }
    if (shape.size() == 2) {
        return shape[0];
    }
    return 0;
}

std::int64_t TracePlotSnapshot::sample_count() const {
    if (shape.size() == 1) {
        return shape[0];
    }
    if (shape.size() == 2) {
        return shape[1];
    }
    return 0;
}

const float *TracePlotSnapshot::trace_data(std::int64_t trace_index) const {
    if (trace_index < 0 || trace_index >= trace_count()) {
        throw std::out_of_range("TracePlotSnapshot trace index is out of range");
    }

    return values.data() + static_cast<std::size_t>(trace_index * sample_count());
}

std::string_view to_string(PlotBackend backend) {
    switch (backend) {
    case PlotBackend::Auto:
        return "auto";
    case PlotBackend::OpenGL3:
        return "opengl3";
    case PlotBackend::Vulkan:
        return "vulkan";
    }

    return "unknown";
}

std::string_view to_string(TracePlotLayout layout) {
    switch (layout) {
    case TracePlotLayout::Overlay:
        return "overlay";
    case TracePlotLayout::Stacked:
        return "stacked";
    }

    return "unknown";
}

std::string_view to_string(TracePlotXAxis axis) {
    switch (axis) {
    case TracePlotXAxis::Sample:
        return "sample";
    case TracePlotXAxis::TimeUs:
        return "time_us";
    }

    return "unknown";
}

PlotBackend parse_plot_backend(std::string_view text) {
    const auto lowered = lower_string(text);
    if (lowered == "auto") {
        return PlotBackend::Auto;
    }
    if (lowered == "opengl3") {
        return PlotBackend::OpenGL3;
    }
    if (lowered == "vulkan") {
        return PlotBackend::Vulkan;
    }

    throw std::invalid_argument("unknown plot backend");
}

TracePlotLayout parse_trace_plot_layout(std::string_view text) {
    const auto lowered = lower_string(text);
    if (lowered == "overlay") {
        return TracePlotLayout::Overlay;
    }
    if (lowered == "stacked") {
        return TracePlotLayout::Stacked;
    }

    throw std::invalid_argument("unknown TracePlot layout");
}

TracePlotXAxis parse_trace_plot_x_axis(std::string_view text) {
    const auto lowered = lower_string(text);
    if (lowered == "sample") {
        return TracePlotXAxis::Sample;
    }
    if (lowered == "time_us") {
        return TracePlotXAxis::TimeUs;
    }

    throw std::invalid_argument("unknown TracePlot x_axis");
}

TracePlotAnnotationMarker parse_trace_plot_annotation_marker(std::string_view text) {
    const auto lowered = lower_string(text);
    if (lowered == "square") {
        return TracePlotAnnotationMarker::Square;
    }
    if (lowered == "x" || lowered == "cross") {
        return TracePlotAnnotationMarker::Cross;
    }
    // "circle" and any unrecognized/empty value default to a filled circle.
    return TracePlotAnnotationMarker::Circle;
}

double trace_plot_annotation_y_from_norm(double norm_value, double lower, double higher) {
    if (!std::isfinite(norm_value) || norm_value < -1.0 || norm_value > 1.0) {
        throw std::invalid_argument("TracePlot annotation norm_value must be finite and between -1 and 1");
    }
    if (!std::isfinite(lower) || !std::isfinite(higher) || lower > higher) {
        throw std::invalid_argument("TracePlot annotation y range must be finite and ordered");
    }
    if (lower == higher) {
        return lower;
    }

    const auto normalized = (norm_value + 1.0) * 0.5;
    if (lower < 0.0 && higher > 0.0) {
        return norm_value * std::max(std::abs(lower), std::abs(higher));
    }
    if (higher <= 0.0) {
        return higher + normalized * (lower - higher);
    }
    return lower + normalized * (higher - lower);
}

std::pair<double, double> trace_plot_centered_y_range(double lower, double higher) {
    if (!std::isfinite(lower) || !std::isfinite(higher) || lower > higher) {
        throw std::invalid_argument("TracePlot centered y range must be finite and ordered");
    }

    const auto max_abs = std::max(std::abs(lower), std::abs(higher));
    if (max_abs == 0.0) {
        return {-1.0, 1.0};
    }

    return {-max_abs, max_abs};
}

std::uint64_t PlotRuntime::add_trace(TracePlotSnapshot snapshot) {
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

bool PlotRuntime::refresh_trace_display(const TracePlotSnapshot &update) {
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

void PlotRuntime::append_score_points(std::string element_name, std::string group, std::string title,
                                      std::string x_label, const std::vector<ScorePointUpdate> &updates) {
    const auto lock = std::scoped_lock(mutex_);
    if (group.empty()) {
        group = "default";
    }

    ScoreSnapshot *snapshot = nullptr;
    for (auto &existing : score_snapshots_) {
        if (existing.element_name == element_name) {
            snapshot = &existing;
            break;
        }
    }
    if (snapshot == nullptr) {
        ScoreSnapshot fresh;
        fresh.id = next_snapshot_id_++;
        fresh.element_name = std::move(element_name);
        snapshot = &score_snapshots_.emplace_back(std::move(fresh));
    }
    // Presentation is refreshed each call (like a ui-control change).
    snapshot->group = std::move(group);
    snapshot->title = std::move(title);
    snapshot->x_label = std::move(x_label);

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

        ScoreSeries *series = nullptr;
        for (auto &candidate : panel->series) {
            if (candidate.label == update.series) {
                series = &candidate;
                break;
            }
        }
        if (series == nullptr) {
            series = &panel->series.emplace_back(ScoreSeries{.label = update.series, .color = update.color});
        }
        series->points.push_back(update.point);
    }
}

bool PlotRuntime::has_sessions() const {
    const auto lock = std::scoped_lock(mutex_);
    return !trace_snapshots_.empty() || !score_snapshots_.empty();
}

const std::vector<TracePlotSnapshot> &PlotRuntime::trace_snapshots() const {
    const auto lock = std::scoped_lock(mutex_);
    return trace_snapshots_;
}

const std::vector<ScoreSnapshot> &PlotRuntime::score_snapshots() const {
    const auto lock = std::scoped_lock(mutex_);
    return score_snapshots_;
}

void PlotRuntime::clear() {
    const auto lock = std::scoped_lock(mutex_);
    trace_snapshots_.clear();
    score_snapshots_.clear();
    // Reset the id counter so a re-registered plot reuses ids after a Stop/Start
    // cycle; otherwise auto palette colors (keyed by id) would shift each cycle.
    // Every id-keyed map below is cleared so reused ids carry no stale state.
    next_snapshot_id_ = 1;
    trace_indices_.clear();
    trace_follow_latest_.clear();
    group_trace_locks_.clear();
    trace_display_states_.clear();
    group_control_windows_.clear();
    axis_views_.clear();
    score_panel_heights_.clear();
}

int &PlotRuntime::mutable_trace_index(std::uint64_t snapshot_id, int initial_value) {
    const auto lock = std::scoped_lock(mutex_);
    return trace_indices_.try_emplace(snapshot_id, initial_value).first->second;
}

bool &PlotRuntime::mutable_trace_follow_latest(std::uint64_t snapshot_id, bool initial_value) {
    const auto lock = std::scoped_lock(mutex_);
    return trace_follow_latest_.try_emplace(snapshot_id, initial_value).first->second;
}

void PlotRuntime::set_streaming(bool streaming) {
    const auto lock = std::scoped_lock(mutex_);
    streaming_ = streaming;
}

bool PlotRuntime::streaming() const {
    const auto lock = std::scoped_lock(mutex_);
    return streaming_;
}

bool &PlotRuntime::mutable_group_trace_lock(std::string_view group, bool initial_value) {
    const auto lock = std::scoped_lock(mutex_);
    return group_trace_locks_.try_emplace(std::string(group), initial_value).first->second;
}

TracePlotDisplayState &PlotRuntime::mutable_trace_display_state(const TracePlotSnapshot &snapshot) {
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

bool &PlotRuntime::mutable_group_controls_open(std::string_view group) {
    const auto lock = std::scoped_lock(mutex_);
    return group_control_windows_.try_emplace(std::string(group), false).first->second;
}

PlotAxisView &PlotRuntime::mutable_axis_view(std::uint64_t panel_id) {
    const auto lock = std::scoped_lock(mutex_);
    return axis_views_[panel_id];
}

float &PlotRuntime::mutable_score_panel_height(const std::string &key, float initial_value) {
    const auto lock = std::scoped_lock(mutex_);
    return score_panel_heights_.try_emplace(key, initial_value).first->second;
}

void PlotRuntime::set_trace_index_listener(std::function<void(std::string_view, int)> listener) {
    const auto lock = std::scoped_lock(mutex_);
    trace_index_listener_ = std::move(listener);
}

void PlotRuntime::set_x_axis_listener(std::function<void(std::string_view, std::string_view)> listener) {
    const auto lock = std::scoped_lock(mutex_);
    x_axis_listener_ = std::move(listener);
}

void PlotRuntime::notify_x_axis(std::string_view element_name, std::string_view x_axis) {
    std::function<void(std::string_view, std::string_view)> listener;
    {
        const auto lock = std::scoped_lock(mutex_);
        listener = x_axis_listener_;
    }
    if (listener && !element_name.empty()) {
        listener(element_name, x_axis);
    }
}

void PlotRuntime::notify_trace_index(std::string_view element_name, int trace_index) {
    std::function<void(std::string_view, int)> listener;
    {
        const auto lock = std::scoped_lock(mutex_);
        listener = trace_index_listener_;
    }
    if (listener && !element_name.empty()) {
        listener(element_name, trace_index);
    }
}

std::recursive_mutex &PlotRuntime::mutex() const { return mutex_; }

} // namespace leakflow::plot
