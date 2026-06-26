#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plugins/plot/descriptor_catalog.hpp"
#include "leakflow/plugins/plot/plot_elements.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

bool expect_near(double actual, double expected, const char* message)
{
    if (std::abs(actual - expected) > 1.0e-5) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
        return false;
    }

    return true;
}

template <typename Exception, typename Function> bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }

    return false;
}

} // namespace

int main()
{
    namespace plot_plugin = leakflow::plugins::plot;

    if (!expect_near(leakflow::plot::trace_plot_annotation_y_from_norm(0.5, -2.0, 1.0), 1.0,
            "TracePlot mixed-sign normalized annotation y was wrong")) {
        return 1;
    }
    if (!expect_near(leakflow::plot::trace_plot_annotation_y_from_norm(0.0, -0.25, 1.0), 0.0,
            "TracePlot mixed-sign normalized annotation center was wrong")) {
        return 1;
    }
    if (!expect_near(leakflow::plot::trace_plot_annotation_y_from_norm(0.0, 2.0, 6.0), 4.0,
            "TracePlot non-negative normalized annotation y was wrong")) {
        return 1;
    }
    if (!expect_near(leakflow::plot::trace_plot_annotation_y_from_norm(0.0, -6.0, -2.0), -4.0,
            "TracePlot non-positive normalized annotation y was wrong")) {
        return 1;
    }
    if (!expect_near(leakflow::plot::trace_plot_annotation_y_from_norm(-1.0, -6.0, -2.0), -2.0,
            "TracePlot non-positive normalized annotation high endpoint was wrong")) {
        return 1;
    }
    const auto centered_mixed = leakflow::plot::trace_plot_centered_y_range(-0.5, 2.0);
    if (!expect_near(centered_mixed.first, -2.0, "TracePlot centered mixed lower limit was wrong")) {
        return 1;
    }
    if (!expect_near(centered_mixed.second, 2.0, "TracePlot centered mixed upper limit was wrong")) {
        return 1;
    }
    const auto centered_zero = leakflow::plot::trace_plot_centered_y_range(0.0, 0.0);
    if (!expect_near(centered_zero.first, -1.0, "TracePlot centered zero lower fallback was wrong")) {
        return 1;
    }
    if (!expect_near(centered_zero.second, 1.0, "TracePlot centered zero upper fallback was wrong")) {
        return 1;
    }

    auto runtime = std::make_shared<leakflow::plot::PlotRuntime>();
    plot_plugin::TracePlot trace_plot(runtime, "plot");
    trace_plot.set_property("group", std::string("aes"));
    trace_plot.set_property("label", std::string("traces"));
    trace_plot.set_property("title", std::string("AES traces"));
    trace_plot.set_property("layout", std::string("overlay"));
    trace_plot.set_property("color", *leakflow::parse_color("rgba(255,128,0,0.25)"));
    trace_plot.set_property("trace_index", std::int64_t{1});
    trace_plot.set_property("order", std::int64_t{4});
    trace_plot.set_property("x_axis", std::string("time_us"));

    auto payload = std::make_shared<leakflow::base::TorchTensorPayload>(
        torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3}));
    leakflow::Buffer input(payload->caps());
    input.set_metadata(leakflow::plot::sample_rate_metadata_key, "1000000");
    input.set_payload(payload);

    auto output = trace_plot.process(input);
    if (!expect(output.has_value(), "TracePlot did not pass the buffer through")) {
        return 1;
    }
    if (!expect(output->payload() == input.payload(), "TracePlot did not preserve the payload handle")) {
        return 1;
    }
    if (!expect(runtime->has_sessions(), "TracePlot did not register a plot session")) {
        return 1;
    }
    if (!expect(runtime->trace_snapshots().size() == 1, "TracePlot registered wrong session count")) {
        return 1;
    }

    const auto& snapshot = runtime->trace_snapshots().front();
    if (!expect(snapshot.group == "aes", "TracePlot snapshot group was wrong")) {
        return 1;
    }
    if (!expect(snapshot.element_name == "plot", "TracePlot snapshot element name was wrong")) {
        return 1;
    }
    if (!expect(snapshot.label == "traces", "TracePlot snapshot label was wrong")) {
        return 1;
    }
    if (!expect(snapshot.title == "AES traces", "TracePlot snapshot title was wrong")) {
        return 1;
    }
    if (!expect(snapshot.layout == leakflow::plot::TracePlotLayout::Overlay, "TracePlot snapshot layout was wrong")) {
        return 1;
    }
    if (!expect(snapshot.color.has_value(), "TracePlot snapshot color was not set")) {
        return 1;
    }
    if (!expect_near(snapshot.color->red, 1.0, "TracePlot snapshot red color component was wrong")) {
        return 1;
    }
    if (!expect_near(snapshot.color->green, 128.0 / 255.0, "TracePlot snapshot green color component was wrong")) {
        return 1;
    }
    if (!expect_near(snapshot.color->blue, 0.0, "TracePlot snapshot blue color component was wrong")) {
        return 1;
    }
    if (!expect(snapshot.x_axis == leakflow::plot::TracePlotXAxis::TimeUs, "TracePlot snapshot x axis was wrong")) {
        return 1;
    }
    if (!expect(snapshot.sample_rate_hz && *snapshot.sample_rate_hz == 1000000.0,
                "TracePlot did not read sample_rate_hz metadata")) {
        return 1;
    }
    if (!expect(snapshot.rank() == 2, "TracePlot snapshot rank was wrong")) {
        return 1;
    }
    if (!expect(snapshot.trace_count() == 2, "TracePlot snapshot trace count was wrong")) {
        return 1;
    }
    if (!expect(snapshot.sample_count() == 3, "TracePlot snapshot sample count was wrong")) {
        return 1;
    }
    if (!expect(snapshot.initial_trace_index == 1, "TracePlot snapshot initial trace index was wrong")) {
        return 1;
    }
    if (!expect(snapshot.order == 4, "TracePlot snapshot order was wrong")) {
        return 1;
    }
    if (!expect(snapshot.center0, "TracePlot center0 should default to true")) {
        return 1;
    }
    if (!expect(snapshot.values == std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F},
                "TracePlot snapshot values were wrong")) {
        return 1;
    }

    auto uncentered_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
    plot_plugin::TracePlot uncentered_plot(uncentered_runtime, "uncentered_plot");
    uncentered_plot.set_property("center0", false);
    (void)uncentered_plot.process(input);
    if (!expect(!uncentered_runtime->trace_snapshots().front().center0,
            "TracePlot center0=false property was not captured")) {
        return 1;
    }

    auto& display_state = runtime->mutable_trace_display_state(snapshot);
    if (!expect(display_state.order == 4, "TracePlot display order was wrong")) {
        return 1;
    }
    if (!expect_near(display_state.color.red, 1.0, "TracePlot display red color component was wrong")) {
        return 1;
    }
    if (!expect_near(display_state.color.green, 128.0 / 255.0,
                     "TracePlot display green color component was wrong")) {
        return 1;
    }
    if (!expect_near(display_state.color.blue, 0.0, "TracePlot display blue color component was wrong")) {
        return 1;
    }
    if (!expect_near(display_state.alpha, 0.25, "TracePlot display alpha was wrong")) {
        return 1;
    }
    display_state.order = 0;
    display_state.alpha = 0.5F;
    if (!expect(runtime->mutable_trace_display_state(snapshot).order == 0,
                "TracePlot display order was not mutable")) {
        return 1;
    }
    if (!expect(runtime->mutable_trace_display_state(snapshot).alpha == 0.5F,
                "TracePlot display alpha was not mutable")) {
        return 1;
    }
    auto& controls_open = runtime->mutable_group_controls_open("aes");
    controls_open = true;
    if (!expect(runtime->mutable_group_controls_open("aes"), "TracePlot group controls state was not mutable")) {
        return 1;
    }
    // The grouped trace-slider lock is a group-level UI toggle (no element
    // property); it starts unlocked and is mutable.
    if (!expect(!runtime->mutable_group_trace_lock("aes", false),
            "TracePlot group trace lock should start unlocked")) {
        return 1;
    }
    auto& trace_lock = runtime->mutable_group_trace_lock("aes", false);
    trace_lock = true;
    if (!expect(runtime->mutable_group_trace_lock("aes", false),
            "TracePlot group trace lock state was not mutable")) {
        return 1;
    }
    trace_lock = false;

    auto annotated_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
    plot_plugin::TracePlot annotated_plot(annotated_runtime, "annotated_plot");
    auto annotation_payload = std::make_shared<leakflow::base::PlotAnnotationPayload>(
        std::vector<leakflow::base::PlotAnnotation>{
            leakflow::base::PlotAnnotation{
                .sample_index = 1,
                .value = 0.875,
                .fields = {{"key byte", "3"}, {"target", "HW(y)"}, {"correlation", "0.875"}},
                .label = "byte_3.HW(y)",
                .text = "byte_3.HW(y): 0.875",
                .kind = "poi",
                .target_index = 1,
            },
            leakflow::base::PlotAnnotation{
                .sample_index = 1,
                .norm_value = -0.5,
                .fields = {{"key byte", "5"}, {"target", "HW(m)"}, {"correlation", "-0.500"}},
                .label = "byte_5.HW(m)",
                .text = "byte_5.HW(m): -0.500",
                .kind = "poi",
                .target_index = 2,
            },
            leakflow::base::PlotAnnotation{
                .sample_index = 2,
                .fields = {{"target", "combined"}},
                .kind = "poi",
            },
        });
    leakflow::Buffer annotation_buffer(annotation_payload->caps());
    annotation_buffer.set_payload(annotation_payload);
    leakflow::ElementInputs annotated_inputs;
    annotated_inputs.emplace("sink", input);
    annotated_inputs.emplace("annotations", annotation_buffer);
    const auto annotated_output = annotated_plot.process_inputs(std::move(annotated_inputs));
    if (!expect(annotated_output.has_value(), "TracePlot with annotations did not pass the buffer through")) {
        return 1;
    }
    if (!expect(annotated_runtime->trace_snapshots().size() == 1,
            "TracePlot with annotations did not register one snapshot")) {
        return 1;
    }
    const auto& annotated_snapshot = annotated_runtime->trace_snapshots().front();
    if (!expect(annotated_snapshot.annotations.size() == 3,
            "TracePlot snapshot annotation count was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[0].sample_index == 1,
            "TracePlot annotation sample index was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[0].value.has_value(),
            "TracePlot annotation value was not set")) {
        return 1;
    }
    if (!expect_near(*annotated_snapshot.annotations[0].value, 0.875,
            "TracePlot annotation value was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[0].fields.size() == 3,
            "TracePlot annotation ordered field count was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[0].fields[0].first == "key byte"
                && annotated_snapshot.annotations[0].fields[0].second == "3",
            "TracePlot annotation ordered key byte field was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[0].fields[1].first == "target"
                && annotated_snapshot.annotations[0].fields[1].second == "HW(y)",
            "TracePlot annotation ordered target field was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[0].label == "byte_3.HW(y)",
            "TracePlot annotation label was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[0].target_index
                && *annotated_snapshot.annotations[0].target_index == 1,
            "TracePlot annotation target index was wrong")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[1].sample_index == 1,
            "TracePlot did not preserve duplicate annotation indexes")) {
        return 1;
    }
    if (!expect(!annotated_snapshot.annotations[1].value,
            "TracePlot normalized annotation should not have exact value")) {
        return 1;
    }
    if (!expect(annotated_snapshot.annotations[1].norm_value
                && *annotated_snapshot.annotations[1].norm_value == -0.5,
            "TracePlot duplicate annotation norm_value was wrong")) {
        return 1;
    }
    if (!expect(!annotated_snapshot.annotations[2].value && !annotated_snapshot.annotations[2].norm_value,
            "TracePlot top annotation should not have y values")) {
        return 1;
    }

    // Line alpha is now part of the color (rgba); a color without an explicit
    // alpha is opaque (1.0).
    auto expect_color_property = [&payload](const std::string& color, double expected_red, double expected_green,
                                            double expected_blue, double expected_alpha) {
        auto color_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot color_plot(color_runtime, "color_plot");
        color_plot.set_property("color", *leakflow::parse_color(color));

        leakflow::Buffer color_input(payload->caps());
        color_input.set_payload(payload);
        (void)color_plot.process(color_input);

        const auto& color_snapshot = color_runtime->trace_snapshots().front();
        const auto& color_state = color_runtime->mutable_trace_display_state(color_snapshot);
        return expect(color_snapshot.color.has_value(), "TracePlot color property did not set snapshot color") &&
               expect_near(color_state.color.red, expected_red, "TracePlot color property red component was wrong") &&
               expect_near(color_state.color.green, expected_green,
                           "TracePlot color property green component was wrong") &&
               expect_near(color_state.color.blue, expected_blue,
                           "TracePlot color property blue component was wrong") &&
               expect_near(color_state.alpha, expected_alpha, "TracePlot color property alpha was wrong");
    };

    if (!expect_color_property("rgb(0,128,255)", 0.0, 128.0 / 255.0, 1.0, 1.0)) {
        return 1;
    }
    if (!expect_color_property("#336699cc", 0x33 / 255.0, 0x66 / 255.0, 0x99 / 255.0, 0xCC / 255.0)) {
        return 1;
    }
    if (!expect_color_property("blue", 0.0, 0.0, 1.0, 1.0)) {
        return 1;
    }
    if (!expect_color_property("rgba(255,0,0,0.2)", 1.0, 0.0, 0.0, 0.2)) {
        return 1;
    }

    plot_plugin::TracePlot bad_dtype_plot(std::make_shared<leakflow::plot::PlotRuntime>(), "bad_dtype");
    auto bad_payload = std::make_shared<leakflow::base::TorchTensorPayload>(
        torch::arange(0, 3, torch::TensorOptions().dtype(torch::kFloat64)));
    leakflow::Buffer bad_input(bad_payload->caps());
    bad_input.set_payload(bad_payload);
    if (!expect(throws_exception<std::invalid_argument>(
                    [&bad_dtype_plot, &bad_input] { (void)bad_dtype_plot.process(bad_input); }),
                "TracePlot accepted non-float32 input")) {
        return 1;
    }

    const auto descriptors = plot_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "plot plugin descriptor count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].name == "leakflow_plugins_plot", "plot plugin descriptor name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 1, "plot element descriptor count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].type_name == "TracePlot", "TracePlot descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].input_pads.size() == 2,
            "TracePlot descriptor input pad count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].input_pads[1].name() == "annotations",
            "TracePlot descriptor annotation pad name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].input_pads[1].presence() == leakflow::PadPresence::Optional,
            "TracePlot descriptor annotation pad was not optional")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].output_pads.empty(),
            "TracePlot descriptor still declared an output pad")) {
        return 1;
    }
    if (!expect(plot_plugin::find_plugin_descriptor("leakflow_plugins_plot") != nullptr,
                "plot plugin descriptor was not findable")) {
        return 1;
    }

    return 0;
}
