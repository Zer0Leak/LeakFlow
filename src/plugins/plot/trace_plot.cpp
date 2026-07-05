#include "leakflow/plugins/plot/trace_plot.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::plot {
namespace {

// Default line alpha when a color carries none (alpha assumed 255 = opaque).
constexpr auto default_trace_alpha = 1.0;

// update_mode resolution. auto follows liveness; the resolved value is mirrored
// into the read-only active_update_mode property for the UI and used to decide
// whether each buffer appends a new snapshot (accumulate) or replaces in place.
[[nodiscard]] bool resolve_accumulate_for(std::string_view update_mode, bool live_driven)
{
    if (update_mode == "accumulate") {
        return true;
    }
    if (update_mode == "replace") {
        return false;
    }
    if (update_mode == "auto") {
        return live_driven;
    }
    throw std::invalid_argument("TracePlot update_mode must be auto, accumulate, or replace");
}

// annotation_update_mode resolution. auto follows the resolved trace mode (today's
// coupled behavior); accumulate/replace override it so annotations can be pinned
// per-trace or kept global independently of whether the traces pile up.
[[nodiscard]] bool resolve_annotations_accumulate_for(std::string_view annotation_update_mode, bool trace_accumulate)
{
    if (annotation_update_mode == "accumulate") {
        return true;
    }
    if (annotation_update_mode == "replace") {
        return false;
    }
    if (annotation_update_mode == "auto") {
        return trace_accumulate;
    }
    throw std::invalid_argument("TracePlot annotation_update_mode must be auto, accumulate, or replace");
}

[[nodiscard]] Caps trace_plot_sink_caps()
{
    Caps caps(leakflow::base::torch_tensor_caps_type);
    caps.set_param(leakflow::base::caps_param_dtype, "float32");
    caps.set_param(leakflow::base::caps_param_device, leakflow::base::cpu_device_caps_value);
    return caps;
}

[[nodiscard]] Caps trace_plot_annotations_caps()
{
    return Caps(leakflow::base::plot_annotation_caps_type);
}

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] std::string lower_string(std::string_view text)
{
    std::string lowered;
    lowered.reserve(text.size());
    for (const auto character : text) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    return lowered;
}

[[nodiscard]] std::string trim_to_string(std::string_view text)
{
    const auto begin = text.find_first_not_of(" \t\n\r");
    if (begin == std::string_view::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\n\r");
    return std::string(text.substr(begin, end - begin + 1));
}

[[nodiscard]] std::vector<std::string> split_comma_list(std::string_view text)
{
    std::vector<std::string> values;
    std::size_t begin = 0;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        if (index == text.size() || text[index] == ',') {
            values.push_back(trim_to_string(text.substr(begin, index - begin)));
            begin = index + 1;
        }
    }
    return values;
}

[[nodiscard]] std::optional<std::vector<std::int64_t>> parse_metadata_int_list(
    std::string_view text,
    std::int64_t expected_count)
{
    const auto trimmed = trim_to_string(text);
    if (trimmed.empty() || trimmed.front() != '[' || trimmed.back() != ']') {
        return std::nullopt;
    }

    const auto body = std::string_view(trimmed).substr(1, trimmed.size() - 2);
    std::vector<std::int64_t> values;
    if (!trim_to_string(body).empty()) {
        for (const auto& part : split_comma_list(body)) {
            std::size_t parsed = 0;
            std::int64_t value = 0;
            try {
                value = std::stoll(part, &parsed, 10);
            } catch (const std::exception&) {
                return std::nullopt;
            }
            if (parsed != part.size()) {
                return std::nullopt;
            }
            values.push_back(value);
        }
    }
    if (values.size() != static_cast<std::size_t>(expected_count)) {
        return std::nullopt;
    }
    return values;
}

[[nodiscard]] double double_property_or(const Element& element, std::string_view name, double fallback)
{
    if (const auto value = element.property_as<double>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] std::int64_t integer_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] bool bool_property_or(const Element& element, std::string_view name, bool fallback)
{
    if (const auto value = element.property_as<bool>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] std::vector<std::int64_t> tensor_shape(const leakflow::base::TorchTensorPayload& payload)
{
    const auto shape = payload.shape();
    return {shape.begin(), shape.end()};
}

[[nodiscard]] std::vector<float> snapshot_values(const torch::Tensor& tensor)
{
    auto contiguous = tensor.contiguous();
    auto flat = contiguous.reshape({-1});

    std::vector<float> values(static_cast<std::size_t>(flat.numel()));
    std::memcpy(values.data(), flat.data_ptr<float>(), values.size() * sizeof(float));
    return values;
}

[[nodiscard]] std::optional<double> metadata_sample_rate(const Buffer& buffer)
{
    if (!buffer.has_metadata(leakflow::plot::sample_rate_metadata_key)) {
        return std::nullopt;
    }

    const auto& text = buffer.metadata(leakflow::plot::sample_rate_metadata_key);
    std::size_t consumed = 0;
    const auto value = std::stod(text, &consumed);
    if (consumed != text.size() || value <= 0.0) {
        throw std::invalid_argument("TracePlot sample_rate_hz metadata must be a positive number");
    }

    return value;
}

[[nodiscard]] std::optional<double> sample_rate_for(const Element& element, const Buffer& buffer)
{
    const auto property_value = double_property_or(element, "sample_rate_hz", 0.0);
    if (property_value > 0.0) {
        return property_value;
    }

    return metadata_sample_rate(buffer);
}

[[nodiscard]] std::vector<leakflow::plot::TracePlotAnnotation> snapshot_annotations(
    const leakflow::base::PlotAnnotationPayload* payload)
{
    std::vector<leakflow::plot::TracePlotAnnotation> annotations;
    if (payload == nullptr) {
        return annotations;
    }

    annotations.reserve(payload->annotation_count());
    for (const auto& annotation : payload->annotations()) {
        annotations.push_back(leakflow::plot::TracePlotAnnotation{
            .sample_index = annotation.sample_index,
            .value = annotation.value,
            .norm_value = annotation.norm_value,
            .fields = annotation.fields,
            .label = annotation.label,
            .text = annotation.text,
            .kind = annotation.kind,
            .target_index = annotation.target_index,
            .marker = leakflow::plot::parse_trace_plot_annotation_marker(annotation.marker),
        });
    }
    return annotations;
}

[[nodiscard]] bool first_tensor_axis_is_attack_unit(const Buffer& buffer)
{
    if (!buffer.has_metadata("tensor.axes")) {
        return false;
    }
    const auto axes = split_comma_list(buffer.metadata("tensor.axes"));
    return !axes.empty() && lower_string(axes.front()) == "attack_unit";
}

[[nodiscard]] std::optional<std::vector<std::int64_t>> trace_context_values_from_metadata(
    const Buffer& buffer,
    std::int64_t trace_count)
{
    if (!buffer.has_metadata("attack.unit.indexes")) {
        return std::nullopt;
    }
    return parse_metadata_int_list(buffer.metadata("attack.unit.indexes"), trace_count);
}

void resolve_trace_context(const Element& element, const Buffer& buffer, leakflow::plot::TracePlotSnapshot& snapshot)
{
    const auto configured_label = trim_to_string(string_property_or(element, "trace_context_label", ""));
    const auto values = trace_context_values_from_metadata(buffer, snapshot.trace_count());
    const auto has_unit_context = first_tensor_axis_is_attack_unit(buffer) || values.has_value();
    snapshot.trace_context_label = configured_label.empty()
        ? (has_unit_context ? "unit" : "trace")
        : configured_label;
    snapshot.trace_context_values = lower_string(snapshot.trace_context_label) == "unit" && values
        ? *values
        : std::vector<std::int64_t>{};
}

// Assign the presentation fields (everything except captured tensor data) from the
// element's properties. Shared by snapshot_for (capture) and the live self-apply in
// property_changed, so a ui-control change updates the snapshot the same way. The
// caller passes the resolved sample rate (buffer metadata fallback on capture;
// property-only on self-apply, where there is no buffer).
void assign_display_properties(const Element& element, leakflow::plot::TracePlotSnapshot& snapshot, bool accumulate,
                               std::optional<double> sample_rate_hz)
{
    snapshot.group = string_property_or(element, "group", "default");
    snapshot.element_name = element.name();
    snapshot.label = string_property_or(element, "label", element.name());
    snapshot.title = string_property_or(element, "title", "");
    snapshot.x_label = string_property_or(element, "x_label", "");
    snapshot.y_label = string_property_or(element, "y_label", "leakage");
    snapshot.layout = leakflow::plot::parse_trace_plot_layout(string_property_or(element, "layout", "overlay"));
    snapshot.line_width = double_property_or(element, "line_width", 1.0);
    // In accumulate mode the slider tracks the growing runtime snapshot, not this
    // single incoming buffer (which has few rows). Force a valid in-range index so a
    // scrubbed trace_index property cannot exceed the incoming buffer's trace count.
    snapshot.initial_trace_index = accumulate ? 0 : integer_property_or(element, "trace_index", 0);
    // order < 0 means automatic: keep registration order (resolved at render time).
    snapshot.order = integer_property_or(element, "order", -1);
    snapshot.trace_context_label = trim_to_string(string_property_or(element, "trace_context_label", ""));
    snapshot.center0 = bool_property_or(element, "center0", true);
    snapshot.x_axis = leakflow::plot::parse_trace_plot_x_axis(string_property_or(element, "x_axis", "sample"));
    snapshot.sample_rate_hz = sample_rate_hz;
    // Line alpha is part of the color now: opaque (255) unless the color carries
    // an explicit alpha. An unset color picks an automatic palette color (opaque).
    snapshot.alpha = default_trace_alpha;
    if (const auto color = element.property_as<leakflow::Color>("color"); color) {
        snapshot.color = leakflow::plot::TracePlotColor{color->r, color->g, color->b};
        snapshot.alpha = static_cast<double>(color->a);
    }
}

[[nodiscard]] leakflow::plot::TracePlotSnapshot snapshot_for(const Element& element, const Buffer& buffer,
                                                             const leakflow::base::TorchTensorPayload& payload,
                                                             const leakflow::base::PlotAnnotationPayload* annotations,
                                                             bool accumulate)
{
    if (payload.dtype() != torch::kFloat32) {
        throw std::invalid_argument("TracePlot requires float32 TorchTensorPayload input");
    }
    if (!payload.is_cpu()) {
        throw std::invalid_argument("TracePlot requires CPU TorchTensorPayload input");
    }
    if (payload.rank() != 1 && payload.rank() != 2) {
        throw std::invalid_argument("TracePlot requires rank 1 or rank 2 TorchTensorPayload input");
    }

    leakflow::plot::TracePlotSnapshot snapshot;
    assign_display_properties(element, snapshot, accumulate, sample_rate_for(element, buffer));
    snapshot.shape = tensor_shape(payload);
    snapshot.values = snapshot_values(payload.tensor());
    resolve_trace_context(element, buffer, snapshot);
    snapshot.annotations = snapshot_annotations(annotations);
    snapshot.accumulate = accumulate;
    snapshot.annotations_accumulate = resolve_annotations_accumulate_for(
        string_property_or(element, "annotation_update_mode", "auto"), accumulate);
    return snapshot;
}

void warn_time_us_without_rate(Element& element, bool& warned)
{
    if (warned) {
        return;
    }
    warned = true;
    auto record = element.make_log_record(log::LogLevel::Warning, "element",
        "x_axis=time_us needs a sample rate (the sample_rate_hz property or capture.sample_rate_hz "
        "metadata); none present, falling back to x_axis=sample");
    leakflow::log::write(std::move(record));
}

// x_axis=time_us needs a sample rate (the sample_rate_hz property override or
// capture.sample_rate_hz metadata). With neither, warn once per run and fall back to
// the sample index so the axis is honest about what it shows. Returns whether the
// fallback applied (the caller resets the x_axis property so the control/graph follow).
[[nodiscard]] bool finalize_time_axis(Element& element, leakflow::plot::TracePlotSnapshot& snapshot, bool& warned)
{
    if (snapshot.x_axis != leakflow::plot::TracePlotXAxis::TimeUs) {
        return false;
    }
    if (snapshot.sample_rate_hz) {
        warned = false; // a rate is available; re-arm the warning if it disappears later
        return false;
    }
    snapshot.x_axis = leakflow::plot::TracePlotXAxis::Sample;
    warn_time_us_without_rate(element, warned);
    return true;
}

[[nodiscard]] const Buffer& required_input(const ElementInputs& inputs, std::string_view pad_name)
{
    const auto found = inputs.find(std::string(pad_name));
    if (found == inputs.end() || !found->second) {
        throw std::invalid_argument("TracePlot requires connected input pad " + std::string(pad_name));
    }

    return *found->second;
}

[[nodiscard]] const Buffer* optional_input(const ElementInputs& inputs, std::string_view pad_name)
{
    const auto found = inputs.find(std::string(pad_name));
    if (found == inputs.end() || !found->second) {
        return nullptr;
    }

    return &*found->second;
}

[[nodiscard]] std::shared_ptr<leakflow::base::TorchTensorPayload> trace_payload_for(const Buffer& input)
{
    if (input.caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument("TracePlot requires leakflow/torch-tensor input caps");
    }

    auto payload = input.payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument("TracePlot requires a TorchTensorPayload");
    }

    return payload;
}

[[nodiscard]] std::shared_ptr<leakflow::base::PlotAnnotationPayload> annotation_payload_for(const Buffer& input)
{
    if (input.caps().type() != leakflow::base::plot_annotation_caps_type) {
        throw std::invalid_argument("TracePlot annotations input requires leakflow/plot-annotations caps");
    }

    auto payload = input.payload_as<leakflow::base::PlotAnnotationPayload>();
    if (!payload) {
        throw std::invalid_argument("TracePlot annotations input requires a PlotAnnotationPayload");
    }

    return payload;
}

std::optional<Buffer> capture_trace_snapshot(
    Element& element,
    leakflow::plot::TraceView& view,
    const Buffer& input,
    const leakflow::base::PlotAnnotationPayload* annotations,
    bool accumulate,
    bool& warned_time_us_no_rate)
{
    const auto payload = trace_payload_for(input);

    auto snapshot = snapshot_for(element, input, *payload, annotations, accumulate);
    const bool time_axis_fell_back = finalize_time_axis(element, snapshot, warned_time_us_no_rate);
    const auto shape_text = leakflow::base::shape_caps_value(snapshot.shape.data(), snapshot.shape.size());
    const auto annotation_count = snapshot.annotations.size();
    const auto snapshot_id = view.add_trace(std::move(snapshot));

    auto record = element.make_log_record(log::LogLevel::Debug, "element", "captured TracePlot snapshot");
    record.fields.emplace("snapshot_id", std::to_string(snapshot_id));
    record.fields.emplace("shape", shape_text);
    record.fields.emplace("dtype", "float32");
    record.fields.emplace("device", "cpu");
    record.fields.emplace("annotations", std::to_string(annotation_count));
    leakflow::log::write(std::move(record));

    // Reconcile the property with the effective axis: when time_us fell back to
    // sample, push x_axis=sample back to the element (the session applies it), so the
    // control panel and graph follow the plot instead of staying on time_us. In a
    // headless/non-graph run there is no listener, so this is a no-op.
    if (time_axis_fell_back) {
        view.notify_x_axis(element.name(), leakflow::plot::to_string(leakflow::plot::TracePlotXAxis::Sample));
    }

    return input;
}

} // namespace

ElementDescriptor TracePlot::descriptor()
{
    // TracePlot properties are presentation (ui-control): they change how the
    // already-captured snapshot is drawn, not what flows downstream. TracePlot
    // applies them to its own snapshot in property_changed, so the session does no
    // rerun and the change shows immediately in any player state (Running / Paused /
    // Idle / Stopped). The snapshot is keyed by element name, so updating one
    // TracePlot leaves other grouped/stacked TracePlots untouched.
    const auto display = PropertyEffect{
        .kind = PropertyEffectKind::UiControl,
        .scope = PropertyInvalidationScope::ElementUi,
    };
    return {
        .type_name = "TracePlot",
        .klass = "Sink/Plot/Trace",
        .purpose = "snapshot CPU float32 Torch traces and show them with ImGui/ImPlot",
        .input_pads =
            {
                Pad("sink", PadDirection::Input, trace_plot_sink_caps()),
                Pad("annotations", PadDirection::Input, trace_plot_annotations_caps(), PadPresence::Optional),
            },
        .property_specs =
            {
                PropertySpec("group", std::string("default"), "plot comparison group", "", std::monostate{}, "",
                             display),
                PropertySpec("label", std::string(), "trace label", "", std::monostate{}, "", display),
                PropertySpec("title", std::string(), "plot title", "", std::monostate{}, "", display),
                PropertySpec("x_label", std::string(), "x-axis label override", "", std::monostate{}, "", display),
                PropertySpec("y_label", std::string("leakage"), "y-axis label", "", std::monostate{}, "", display),
                PropertySpec("layout", std::string("overlay"), "group comparison layout", "",
                             StringEnumConstraint{{"overlay", "stacked"}}, "", display),
                // order is live per-snapshot display state controlled by the floating
                // controls window and drag-reorder; it is preserved across reruns, so
                // the property sets the initial value only (no sink-display reprocess).
                // color and trace_index are sink-display so a property/picker edit
                // updates the plot (the plot runtime only syncs them when the property
                // value actually changes, so a data rerun still preserves a manual
                // color/slider tweak). Line alpha is part of the color (rgba); there is
                // no separate alpha property. The grouped trace-slider lock is a
                // group-level UI toggle (the group Controls window and slider
                // right-click), not a per-element property.
                PropertySpec("color", leakflow::Color{1.0F, 1.0F, 1.0F, 1.0F},
                             "line color (null/auto picks a palette color); alpha is the rgba alpha",
                             "", std::monostate{}, "", display, /*optional=*/true),
                PropertySpec("line_width", 1.0, "plot line width", "", DoubleRangeConstraint{0.1, 20.0}, "", display),
                PropertySpec("trace_index", std::int64_t{0}, "initial rank-2 trace index", "",
                             IntRangeConstraint{0, std::numeric_limits<std::int64_t>::max()}, "", display),
                PropertySpec("trace_context_label", std::string(),
                             "rank-2 slider context label; empty auto-derives trace or unit",
                             "", std::monostate{}, "", display),
                PropertySpec("order", std::int64_t{-1}, "group display order; -1 = automatic"),
                PropertySpec("center0", true,
                             "center y-axis at zero with symmetric limits from the maximum absolute leakage",
                             "", std::monostate{}, "", display),
                PropertySpec("x_axis", std::string("sample"), "x-axis mode", "",
                             StringEnumConstraint{{"sample", "time_us"}}, "", display),
                PropertySpec("sample_rate_hz", 0.0,
                             "sample-rate override; zero uses buffer metadata or "
                             "sample index",
                             "Hz", DoubleRangeConstraint{0.0, 1.0e15}, "", display),
                PropertySpec("update_mode", std::string("auto"),
                             "how successive buffers combine: accumulate appends each buffer as a "
                             "new trace row (scrub with the slider), replace shows only the latest, "
                             "auto follows liveness",
                             "", StringEnumConstraint{{"auto", "accumulate", "replace"}}, "", display),
                PropertySpec("active_update_mode", std::string("replace"),
                             "resolved update mode currently selected by update_mode", "",
                             StringEnumConstraint{{"accumulate", "replace"}}, "", PropertyEffect{},
                             /*optional=*/false, /*writable=*/false),
                PropertySpec("annotation_update_mode", std::string("auto"),
                             "how annotations combine, independent of update_mode: replace keeps only the "
                             "latest buffer's markers (global, drawn on any trace -- best for PoIs over "
                             "accumulated traces), accumulate pins each buffer's markers to the rows it "
                             "added (per-trace history), auto follows update_mode",
                             "", StringEnumConstraint{{"auto", "accumulate", "replace"}}, "", display),
                PropertySpec("active_annotation_update_mode", std::string("replace"),
                             "resolved annotation update mode currently selected by annotation_update_mode", "",
                             StringEnumConstraint{{"accumulate", "replace"}}, "", PropertyEffect{},
                             /*optional=*/false, /*writable=*/false),
            },
        .keywords = {"plot", "trace", "imgui", "implot", "torch"},
        .metadata_suggestions = {
            make_element_metadata_descriptor(
                "capture.sample_rate_hz",
                0.0,
                "trace sample rate used when x_axis=time_us and no property override is set",
                {"29454545.454545453"},
                "TracePlot reads this metadata after its sample_rate_hz property override.",
                {},
                {metadata_pad(PadDirection::Input, "sink")}),
            make_element_metadata_descriptor(
                "capture.source",
                std::string(),
                "capture hardware, simulator, or acquisition source shown by surrounding tooling",
                {"ChipWhisperer"},
                {},
                {},
                {metadata_pad(PadDirection::Input, "sink")}),
            make_element_metadata_descriptor(
                "payload.leakage.inverted",
                false,
                "whether leakage values vary inversely (true) or directly (false) with power consumption",
                {"false"},
                {},
                "true or false",
                {metadata_pad(PadDirection::Input, "sink")}),
        },
    };
}

TracePlot::TracePlot(std::string name) : TracePlot(std::make_shared<leakflow::plot::TraceView>(), std::move(name))
{
}

TracePlot::TracePlot(std::shared_ptr<leakflow::plot::TraceView> view, std::string name)
    : Element(std::move(name)), view_(std::move(view))
{
    if (!view_) {
        throw std::invalid_argument("TracePlot requires a TraceView");
    }

    configure_from_descriptor(descriptor());
}

TracePlot::TracePlot(const std::shared_ptr<leakflow::plot::PlotRuntime> &runtime, std::string name)
    : TracePlot(runtime ? runtime->trace_view() : nullptr, std::move(name))
{
}

void TracePlot::start()
{
    update_active_update_mode();
    warned_time_us_no_rate_ = false;
}

bool TracePlot::resolve_accumulate()
{
    const auto accumulate = resolve_accumulate_for(string_property_or(*this, "update_mode", "auto"), is_live_driven());
    set_read_only_property("active_update_mode", std::string(accumulate ? "accumulate" : "replace"));
    const auto annotations_accumulate = resolve_annotations_accumulate_for(
        string_property_or(*this, "annotation_update_mode", "auto"), accumulate);
    set_read_only_property("active_annotation_update_mode",
        std::string(annotations_accumulate ? "accumulate" : "replace"));
    return accumulate;
}

void TracePlot::update_active_update_mode() { (void)resolve_accumulate(); }

void TracePlot::refresh_display(bool force_y_refit)
{
    const auto accumulate = resolve_accumulate_for(string_property_or(*this, "update_mode", "auto"), is_live_driven());
    const auto sample_rate = double_property_or(*this, "sample_rate_hz", 0.0);
    leakflow::plot::TracePlotSnapshot snapshot;
    assign_display_properties(*this, snapshot, accumulate,
        sample_rate > 0.0 ? std::optional<double>(sample_rate) : std::nullopt);
    if (view_->refresh_trace_display(snapshot, force_y_refit)) {
        // The live snapshot had no rate, so x_axis=time_us fell back to sample. Warn
        // once and reset the property so the control panel and graph follow the plot
        // (no rerun; the session applies the reset like any ui-control change).
        warn_time_us_without_rate(*this, warned_time_us_no_rate_);
        view_->notify_x_axis(name(), leakflow::plot::to_string(leakflow::plot::TracePlotXAxis::Sample));
    }
}

void TracePlot::property_changed(std::string_view name)
{
    if (name == "update_mode" || name == "annotation_update_mode") {
        update_active_update_mode();
    }
    // ui-control (presentation) change: apply it to the live snapshot directly so it
    // shows immediately in any player state, with no rerun. No-op before the first
    // buffer (no snapshot registered yet). active_update_mode is read-only and is set
    // through set_read_only_property, which does not call property_changed.
    refresh_display(name == "center0");
}

void TracePlot::live_driven_changed() { update_active_update_mode(); }

std::optional<Buffer> TracePlot::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("TracePlot requires an input buffer");
    }
    return capture_trace_snapshot(*this, *view_, *input, nullptr, resolve_accumulate(), warned_time_us_no_rate_);
}

std::optional<Buffer> TracePlot::process_inputs(ElementInputs inputs)
{
    const auto& trace_input = required_input(inputs, "sink");
    std::shared_ptr<leakflow::base::PlotAnnotationPayload> annotations;
    if (const auto* annotation_input = optional_input(inputs, "annotations")) {
        annotations = annotation_payload_for(*annotation_input);
    }

    return capture_trace_snapshot(*this, *view_, trace_input, annotations.get(), resolve_accumulate(),
        warned_time_us_no_rate_);
}

std::shared_ptr<leakflow::plot::TraceView> TracePlot::trace_view() const
{
    return view_;
}

void TracePlot::set_trace_view(std::shared_ptr<leakflow::plot::TraceView> view)
{
    if (!view) {
        throw std::invalid_argument("TracePlot requires a TraceView");
    }

    view_ = std::move(view);
}

} // namespace leakflow::plugins::plot
