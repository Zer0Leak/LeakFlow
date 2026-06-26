#include "leakflow/plugins/plot/trace_plot.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"

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
        });
    }
    return annotations;
}

[[nodiscard]] leakflow::plot::TracePlotSnapshot snapshot_for(const Element& element, const Buffer& buffer,
                                                             const leakflow::base::TorchTensorPayload& payload,
                                                             const leakflow::base::PlotAnnotationPayload* annotations)
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
    snapshot.group = string_property_or(element, "group", "default");
    snapshot.element_name = element.name();
    snapshot.label = string_property_or(element, "label", element.name());
    snapshot.title = string_property_or(element, "title", "");
    snapshot.x_label = string_property_or(element, "x_label", "");
    snapshot.y_label = string_property_or(element, "y_label", "leakage");
    snapshot.layout = leakflow::plot::parse_trace_plot_layout(string_property_or(element, "layout", "overlay"));
    snapshot.line_width = double_property_or(element, "line_width", 1.0);
    snapshot.initial_trace_index = integer_property_or(element, "trace_index", 0);
    // order < 0 means automatic: keep registration order (resolved at render time).
    snapshot.order = integer_property_or(element, "order", -1);
    snapshot.center0 = bool_property_or(element, "center0", true);
    snapshot.x_axis = leakflow::plot::parse_trace_plot_x_axis(string_property_or(element, "x_axis", "sample"));
    snapshot.sample_rate_hz = sample_rate_for(element, buffer);
    // Line alpha is part of the color now: opaque (255) unless the color carries
    // an explicit alpha. An unset color picks an automatic palette color (opaque).
    snapshot.alpha = default_trace_alpha;
    if (const auto color = element.property_as<leakflow::Color>("color"); color) {
        snapshot.color = leakflow::plot::TracePlotColor{color->r, color->g, color->b};
        snapshot.alpha = static_cast<double>(color->a);
    }
    snapshot.shape = tensor_shape(payload);
    snapshot.values = snapshot_values(payload.tensor());
    snapshot.annotations = snapshot_annotations(annotations);
    return snapshot;
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
    leakflow::plot::PlotRuntime& runtime,
    const Buffer& input,
    const leakflow::base::PlotAnnotationPayload* annotations)
{
    const auto payload = trace_payload_for(input);

    auto snapshot = snapshot_for(element, input, *payload, annotations);
    const auto shape_text = leakflow::base::shape_caps_value(snapshot.shape.data(), snapshot.shape.size());
    const auto annotation_count = snapshot.annotations.size();
    const auto snapshot_id = runtime.add_trace(std::move(snapshot));

    auto record = element.make_log_record(log::LogLevel::Debug, "element", "captured TracePlot snapshot");
    record.fields.emplace("snapshot_id", std::to_string(snapshot_id));
    record.fields.emplace("shape", shape_text);
    record.fields.emplace("dtype", "float32");
    record.fields.emplace("device", "cpu");
    record.fields.emplace("annotations", std::to_string(annotation_count));
    leakflow::log::write(std::move(record));

    return input;
}

} // namespace

ElementDescriptor TracePlot::descriptor()
{
    // TracePlot is a sink: property changes refresh this element's own plot
    // snapshot (sink-display) without rerunning upstream. The snapshot is keyed
    // by element name, so updating one TracePlot leaves other (possibly grouped/
    // stacked) TracePlots, their sliders, legends, and annotations untouched.
    const auto display = PropertyEffect{
        .kind = PropertyEffectKind::SinkDisplay,
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

TracePlot::TracePlot(std::string name) : TracePlot(std::make_shared<leakflow::plot::PlotRuntime>(), std::move(name))
{
}

TracePlot::TracePlot(std::shared_ptr<leakflow::plot::PlotRuntime> runtime, std::string name)
    : Element(std::move(name)), runtime_(std::move(runtime))
{
    if (!runtime_) {
        throw std::invalid_argument("TracePlot requires a PlotRuntime");
    }

    configure_from_descriptor(descriptor());
}

std::optional<Buffer> TracePlot::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("TracePlot requires an input buffer");
    }
    return capture_trace_snapshot(*this, *runtime_, *input, nullptr);
}

std::optional<Buffer> TracePlot::process_inputs(ElementInputs inputs)
{
    const auto& trace_input = required_input(inputs, "sink");
    std::shared_ptr<leakflow::base::PlotAnnotationPayload> annotations;
    if (const auto* annotation_input = optional_input(inputs, "annotations")) {
        annotations = annotation_payload_for(*annotation_input);
    }

    return capture_trace_snapshot(*this, *runtime_, trace_input, annotations.get());
}

std::shared_ptr<leakflow::plot::PlotRuntime> TracePlot::plot_runtime() const
{
    return runtime_;
}

void TracePlot::set_plot_runtime(std::shared_ptr<leakflow::plot::PlotRuntime> runtime)
{
    if (!runtime) {
        throw std::invalid_argument("TracePlot requires a PlotRuntime");
    }

    runtime_ = std::move(runtime);
}

} // namespace leakflow::plugins::plot
