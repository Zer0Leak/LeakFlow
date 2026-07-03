#include "leakflow/plot/plot_runtime.hpp"

#include "leakflow/log/logger.hpp"
#include "leakflow/plot/plot_view.hpp"
#include "leakflow/plot/trace_view.hpp"

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

std::pair<double, double> trace_plot_fitted_y_range(double lower, double higher) {
    if (!std::isfinite(lower) || !std::isfinite(higher) || lower > higher) {
        throw std::invalid_argument("TracePlot fitted y range must be finite and ordered");
    }

    if (lower == higher) {
        const auto padding = std::max(1.0, std::abs(lower) * 0.05);
        return {lower - padding, higher + padding};
    }

    return {lower, higher};
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

PlotRuntime::PlotRuntime() {
    // The built-in trace plot is a registered view like any other; keeping a typed
    // handle lets TracePlot elements and the graph reach it (trace_view()).
    trace_view_ = std::make_shared<TraceView>();
    views_.push_back(trace_view_);
}

void PlotRuntime::add_view(std::shared_ptr<PlotView> view) {
    const auto lock = std::scoped_lock(mutex_);
    if (view) {
        views_.push_back(std::move(view));
    }
}

const std::vector<std::shared_ptr<PlotView>> &PlotRuntime::views() const {
    const auto lock = std::scoped_lock(mutex_);
    return views_;
}

const std::shared_ptr<TraceView> &PlotRuntime::trace_view() const {
    const auto lock = std::scoped_lock(mutex_);
    return trace_view_;
}

bool PlotRuntime::has_sessions() const {
    const auto lock = std::scoped_lock(mutex_);
    for (const auto &view : views_) {
        if (view && !view->empty()) {
            return true;
        }
    }
    return false;
}

void PlotRuntime::clear() {
    const auto lock = std::scoped_lock(mutex_);
    // Every plot type (TraceView, ScoreView, ...) owns and clears its own data.
    for (const auto &view : views_) {
        if (view) {
            view->clear();
        }
    }
}

std::recursive_mutex &PlotRuntime::mutex() const { return mutex_; }

} // namespace leakflow::plot
