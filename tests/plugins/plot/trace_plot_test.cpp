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
    const auto fitted_flat = leakflow::plot::trace_plot_fitted_y_range(5.0, 5.0);
    if (!expect_near(fitted_flat.first, 4.0, "TracePlot fitted flat lower fallback was wrong")) {
        return 1;
    }
    if (!expect_near(fitted_flat.second, 6.0, "TracePlot fitted flat upper fallback was wrong")) {
        return 1;
    }
    const auto fitted_mixed = leakflow::plot::trace_plot_fitted_y_range(-0.5, 2.0);
    if (!expect_near(fitted_mixed.first, -0.5, "TracePlot fitted mixed lower limit was wrong")) {
        return 1;
    }
    if (!expect_near(fitted_mixed.second, 2.0, "TracePlot fitted mixed upper limit was wrong")) {
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
    if (!expect(runtime->trace_view()->trace_snapshots().size() == 1, "TracePlot registered wrong session count")) {
        return 1;
    }

    const auto& snapshot = runtime->trace_view()->trace_snapshots().front();
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
    if (!expect(snapshot.trace_context_label == "trace", "TracePlot context label should default to trace")) {
        return 1;
    }
    if (!expect(snapshot.trace_context_values.empty(), "TracePlot default context values should be implicit")) {
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
    if (!expect(!uncentered_runtime->trace_view()->trace_snapshots().front().center0,
            "TracePlot center0=false property was not captured")) {
        return 1;
    }

    auto unit_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
    plot_plugin::TracePlot unit_plot(unit_runtime, "unit_plot");
    auto unit_payload = std::make_shared<leakflow::base::TorchTensorPayload>(
        torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3}));
    leakflow::Buffer unit_input(unit_payload->caps());
    unit_input.set_metadata("tensor.axes", "attack_unit,sample");
    unit_input.set_metadata("attack.unit.indexes", "[0,3]");
    unit_input.set_payload(unit_payload);
    (void)unit_plot.process(unit_input);
    const auto& unit_snapshot = unit_runtime->trace_view()->trace_snapshots().front();
    if (!expect(unit_snapshot.trace_context_label == "unit",
            "TracePlot should auto-derive unit context from tensor axes")) {
        return 1;
    }
    if (!expect(unit_snapshot.trace_context_values == std::vector<std::int64_t>{0, 3},
            "TracePlot should capture unit context values from metadata")) {
        return 1;
    }

    auto custom_context_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
    plot_plugin::TracePlot custom_context_plot(custom_context_runtime, "custom_context_plot");
    custom_context_plot.set_property("trace_context_label", std::string("row"));
    (void)custom_context_plot.process(unit_input);
    const auto& custom_context_snapshot = custom_context_runtime->trace_view()->trace_snapshots().front();
    if (!expect(custom_context_snapshot.trace_context_label == "row",
            "TracePlot explicit context label was not captured")) {
        return 1;
    }
    if (!expect(custom_context_snapshot.trace_context_values.empty(),
            "TracePlot custom context label should use implicit row indexes")) {
        return 1;
    }

    auto& display_state = runtime->trace_view()->mutable_trace_display_state(snapshot);
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
    if (!expect(runtime->trace_view()->mutable_trace_display_state(snapshot).order == 0,
                "TracePlot display order was not mutable")) {
        return 1;
    }
    if (!expect(runtime->trace_view()->mutable_trace_display_state(snapshot).alpha == 0.5F,
                "TracePlot display alpha was not mutable")) {
        return 1;
    }
    auto& controls_open = runtime->trace_view()->mutable_group_controls_open("aes");
    controls_open = true;
    if (!expect(runtime->trace_view()->mutable_group_controls_open("aes"), "TracePlot group controls state was not mutable")) {
        return 1;
    }
    auto& panel_height = runtime->trace_view()->mutable_panel_height("aes/0/1", 260.0F);
    if (!expect(panel_height == 260.0F, "TracePlot panel height default was wrong")) {
        return 1;
    }
    panel_height = 444.0F;
    if (!expect(runtime->trace_view()->mutable_panel_height("aes/0/1", 260.0F) == 444.0F,
            "TracePlot panel height was not mutable")) {
        return 1;
    }
    // The grouped trace-slider lock is a group-level UI toggle (no element
    // property); it starts unlocked and is mutable.
    if (!expect(!runtime->trace_view()->mutable_group_trace_lock("aes", false),
            "TracePlot group trace lock should start unlocked")) {
        return 1;
    }
    auto& trace_lock = runtime->trace_view()->mutable_group_trace_lock("aes", false);
    trace_lock = true;
    if (!expect(runtime->trace_view()->mutable_group_trace_lock("aes", false),
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
    if (!expect(annotated_runtime->trace_view()->trace_snapshots().size() == 1,
            "TracePlot with annotations did not register one snapshot")) {
        return 1;
    }
    const auto& annotated_snapshot = annotated_runtime->trace_view()->trace_snapshots().front();
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

        const auto& color_snapshot = color_runtime->trace_view()->trace_snapshots().front();
        const auto& color_state = color_runtime->trace_view()->mutable_trace_display_state(color_snapshot);
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

    // update_mode: replace keeps one snapshot updated in place (auto resolves to
    // replace for a non-live-driven element); accumulate grows that one snapshot by
    // appending each buffer as a new trace row, with annotations bound to their row.
    // active_update_mode mirrors the resolved value and is read-only.
    const auto make_trace_buffer = [](std::int64_t samples) {
        auto trace_payload = std::make_shared<leakflow::base::TorchTensorPayload>(
            torch::arange(0, samples, torch::TensorOptions().dtype(torch::kFloat32)));
        leakflow::Buffer buffer(trace_payload->caps());
        buffer.set_payload(trace_payload);
        return buffer;
    };

    {
        auto replace_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot replace_plot(replace_runtime, "replace_plot");
        (void)replace_plot.process(make_trace_buffer(4));
        (void)replace_plot.process(make_trace_buffer(4));
        if (!expect(replace_runtime->trace_view()->trace_snapshots().size() == 1,
                    "TracePlot auto/offline update_mode should replace in place")) {
            return 1;
        }
        if (!expect(replace_plot.property_as<std::string>("active_update_mode") == std::string("replace"),
                    "TracePlot auto/offline active_update_mode should be replace")) {
            return 1;
        }
    }

    {
        auto accumulate_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot accumulate_plot(accumulate_runtime, "accumulate_plot");
        accumulate_plot.set_property("update_mode", std::string("accumulate"));
        (void)accumulate_plot.process(make_trace_buffer(4));
        (void)accumulate_plot.process(make_trace_buffer(4));
        (void)accumulate_plot.process(make_trace_buffer(4));
        if (!expect(accumulate_runtime->trace_view()->trace_snapshots().size() == 1,
                    "TracePlot accumulate should grow one snapshot, not add snapshots")) {
            return 1;
        }
        const auto& accumulated = accumulate_runtime->trace_view()->trace_snapshots().front();
        if (!expect(accumulated.trace_count() == 3, "TracePlot accumulate should append one trace per buffer")) {
            return 1;
        }
        if (!expect(accumulated.rank() == 2, "TracePlot accumulate snapshot should become rank 2")) {
            return 1;
        }
        if (!expect(accumulated.sample_count() == 4, "TracePlot accumulate sample count was wrong")) {
            return 1;
        }
        if (!expect(accumulated.values.size() == 12, "TracePlot accumulate did not concatenate trace rows")) {
            return 1;
        }
        if (!expect(accumulate_plot.property_as<std::string>("active_update_mode") == std::string("accumulate"),
                    "TracePlot accumulate active_update_mode was wrong")) {
            return 1;
        }
    }

    {
        // A scrubbed trace_index (e.g. from the slider) must not make the next
        // single-row accumulate buffer fail validation (initial index out of range).
        auto scrub_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot scrub_plot(scrub_runtime, "scrub_plot");
        scrub_plot.set_property("update_mode", std::string("accumulate"));
        (void)scrub_plot.process(make_trace_buffer(4));
        scrub_plot.set_property("trace_index", std::int64_t{5});
        if (!expect(!throws_exception<std::invalid_argument>([&] { (void)scrub_plot.process(make_trace_buffer(4)); }),
                    "TracePlot accumulate must ignore a scrubbed trace_index for incoming buffers")) {
            return 1;
        }
        if (!expect(scrub_runtime->trace_view()->trace_snapshots().front().trace_count() == 2,
                    "TracePlot accumulate should keep appending after a scrubbed trace_index")) {
            return 1;
        }
    }

    {
        auto history_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot history_plot(history_runtime, "history_plot");
        history_plot.set_property("update_mode", std::string("accumulate"));
        const auto feed_annotated = [&](double value) {
            auto annotation_payload = std::make_shared<leakflow::base::PlotAnnotationPayload>(
                std::vector<leakflow::base::PlotAnnotation>{
                    leakflow::base::PlotAnnotation{.sample_index = 1, .value = value, .kind = "poi"}});
            leakflow::Buffer annotation_buffer(annotation_payload->caps());
            annotation_buffer.set_payload(annotation_payload);
            leakflow::ElementInputs inputs;
            inputs.emplace("sink", make_trace_buffer(4));
            inputs.emplace("annotations", annotation_buffer);
            (void)history_plot.process_inputs(std::move(inputs));
        };
        feed_annotated(0.1);
        feed_annotated(0.2);
        if (!expect(history_runtime->trace_view()->trace_snapshots().size() == 1,
                    "TracePlot accumulate annotations should stay in one snapshot")) {
            return 1;
        }
        const auto& history = history_runtime->trace_view()->trace_snapshots().front();
        if (!expect(history.annotations.size() == 2, "TracePlot accumulate should keep one annotation per trace")) {
            return 1;
        }
        if (!expect(history.annotations[0].trace_row == 0 && history.annotations[1].trace_row == 1,
                    "TracePlot accumulate annotations were not bound to their trace rows")) {
            return 1;
        }
    }

    {
        auto mismatch_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot mismatch_plot(mismatch_runtime, "mismatch_plot");
        mismatch_plot.set_property("update_mode", std::string("accumulate"));
        (void)mismatch_plot.process(make_trace_buffer(4));
        if (!expect(throws_exception<std::invalid_argument>(
                        [&] { (void)mismatch_plot.process(make_trace_buffer(5)); }),
                    "TracePlot accumulate should reject a changed sample count")) {
            return 1;
        }
    }

    {
        plot_plugin::TracePlot read_only_plot(std::make_shared<leakflow::plot::PlotRuntime>(), "read_only_plot");
        if (!expect(throws_exception<std::invalid_argument>([&read_only_plot] {
                        read_only_plot.set_property("active_update_mode", std::string("accumulate"));
                    }),
                    "TracePlot active_update_mode should be read-only")) {
            return 1;
        }
    }

    {
        // Stop clears the runtime; re-registering must reuse ids so the auto palette
        // color (keyed by id) stays stable across a Stop/Start cycle.
        auto cycle_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot cycle_plot(cycle_runtime, "cycle_plot");
        (void)cycle_plot.process(make_trace_buffer(4));
        const auto first_id = cycle_runtime->trace_view()->trace_snapshots().front().id;
        const auto first_color = cycle_runtime->trace_view()->trace_snapshots().front().color;
        cycle_runtime->trace_view()->mutable_panel_height("cycle/0/1", 260.0F) = 512.0F;
        cycle_runtime->clear();
        (void)cycle_plot.process(make_trace_buffer(4));
        const auto second_id = cycle_runtime->trace_view()->trace_snapshots().front().id;
        const auto second_color = cycle_runtime->trace_view()->trace_snapshots().front().color;
        if (!expect(first_id == second_id, "PlotRuntime clear should reset snapshot ids")) {
            return 1;
        }
        if (!expect(first_color.has_value() && second_color.has_value() && *first_color == *second_color,
                    "PlotRuntime clear should keep auto palette colors stable across a Stop/Start cycle")) {
            return 1;
        }
        if (!expect(cycle_runtime->trace_view()->mutable_panel_height("cycle/0/1", 260.0F) == 260.0F,
                    "PlotRuntime clear should reset TracePlot panel heights")) {
            return 1;
        }
    }

    {
        // ui-control self-apply: a presentation property change updates the live
        // snapshot directly (no rerun, no new buffer), so it takes effect in any state.
        auto ui_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot ui_plot(ui_runtime, "ui_plot");
        ui_plot.set_property("center0", false); // before any buffer: no snapshot, must not crash
        (void)ui_plot.process(make_trace_buffer(4));
        auto &axis_view =
            ui_runtime->trace_view()->mutable_axis_view(ui_runtime->trace_view()->trace_snapshots().front().id);
        axis_view.y_fit_initialized = true;
        axis_view.y_user_adjusted = true;
        ui_plot.set_property("center0", true);
        if (!expect(ui_runtime->trace_view()->trace_snapshots().front().center0,
                    "TracePlot center0 change should self-apply to the snapshot without a rerun")) {
            return 1;
        }
        if (!expect(!axis_view.y_fit_initialized && !axis_view.y_user_adjusted,
                    "TracePlot center0 change should request a fresh y-axis fit")) {
            return 1;
        }
        axis_view.y_fit_initialized = true;
        axis_view.y_user_adjusted = true;
        ui_plot.set_property("center0", false);
        if (!expect(!ui_runtime->trace_view()->trace_snapshots().front().center0,
                    "TracePlot center0=false should self-apply to the snapshot")) {
            return 1;
        }
        if (!expect(!axis_view.y_fit_initialized && !axis_view.y_user_adjusted,
                    "TracePlot center0=false should request a fresh y-axis fit")) {
            return 1;
        }
        ui_plot.set_property("title", std::string("updated"));
        if (!expect(ui_runtime->trace_view()->trace_snapshots().front().title == "updated",
                    "TracePlot title change should self-apply to the snapshot")) {
            return 1;
        }
        ui_plot.set_property("trace_context_label", std::string("unit"));
        if (!expect(ui_runtime->trace_view()->trace_snapshots().front().trace_context_label == "unit",
                    "TracePlot context label change should self-apply to the snapshot")) {
            return 1;
        }
        if (!expect(ui_runtime->trace_view()->trace_snapshots().size() == 1,
                    "TracePlot ui-control self-apply should not add a snapshot")) {
            return 1;
        }
    }

    {
        // x_axis=time_us with no sample rate (no property, no capture.sample_rate_hz
        // metadata) falls back to sample.
        auto time_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot time_plot(time_runtime, "time_plot");
        time_plot.set_property("x_axis", std::string("time_us"));
        (void)time_plot.process(make_trace_buffer(4));
        if (!expect(time_runtime->trace_view()->trace_snapshots().front().x_axis == leakflow::plot::TracePlotXAxis::Sample,
                    "TracePlot x_axis=time_us without a sample rate should fall back to sample")) {
            return 1;
        }
    }

    {
        // With capture.sample_rate_hz metadata present, time_us is honored.
        auto rate_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot rate_plot(rate_runtime, "rate_plot");
        rate_plot.set_property("x_axis", std::string("time_us"));
        auto rate_buffer = make_trace_buffer(4);
        rate_buffer.set_metadata(leakflow::plot::sample_rate_metadata_key, "1000000");
        (void)rate_plot.process(rate_buffer);
        if (!expect(rate_runtime->trace_view()->trace_snapshots().front().x_axis == leakflow::plot::TracePlotXAxis::TimeUs,
                    "TracePlot x_axis=time_us with a sample rate should be honored")) {
            return 1;
        }
    }

    {
        // Editing x_axis to time_us (control panel) with no rate must fall back to
        // sample on the live snapshot too (self-apply path), not only at capture.
        auto edit_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot edit_plot(edit_runtime, "edit_plot");
        (void)edit_plot.process(make_trace_buffer(4));
        edit_plot.set_property("x_axis", std::string("time_us"));
        if (!expect(edit_runtime->trace_view()->trace_snapshots().front().x_axis == leakflow::plot::TracePlotXAxis::Sample,
                    "TracePlot x_axis=time_us edit without a rate should fall back to sample on the snapshot")) {
            return 1;
        }
    }

    {
        // Annotation marker hint maps to the snapshot marker shape (square/x/circle).
        if (!expect(leakflow::plot::parse_trace_plot_annotation_marker("square") ==
                            leakflow::plot::TracePlotAnnotationMarker::Square &&
                        leakflow::plot::parse_trace_plot_annotation_marker("x") ==
                            leakflow::plot::TracePlotAnnotationMarker::Cross &&
                        leakflow::plot::parse_trace_plot_annotation_marker("circle") ==
                            leakflow::plot::TracePlotAnnotationMarker::Circle &&
                        leakflow::plot::parse_trace_plot_annotation_marker("") ==
                            leakflow::plot::TracePlotAnnotationMarker::Circle,
                    "parse_trace_plot_annotation_marker mapping was wrong")) {
            return 1;
        }

        auto marker_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot marker_plot(marker_runtime, "marker_plot");
        auto marker_annotations =
            std::make_shared<leakflow::base::PlotAnnotationPayload>(std::vector<leakflow::base::PlotAnnotation>{
                leakflow::base::PlotAnnotation{
                    .sample_index = 1, .norm_value = 0.5, .kind = "attack", .marker = "x"}});
        leakflow::Buffer marker_annotation_buffer(marker_annotations->caps());
        marker_annotation_buffer.set_payload(marker_annotations);
        leakflow::ElementInputs marker_inputs;
        marker_inputs.emplace("sink", make_trace_buffer(4));
        marker_inputs.emplace("annotations", marker_annotation_buffer);
        (void)marker_plot.process_inputs(std::move(marker_inputs));
        if (!expect(marker_runtime->trace_view()->trace_snapshots().front().annotations.front().marker ==
                        leakflow::plot::TracePlotAnnotationMarker::Cross,
                    "TracePlot should map annotation marker=x to Cross on the snapshot")) {
            return 1;
        }
    }

    // annotation_update_mode decouples annotations from the trace update_mode. With
    // update_mode=accumulate the traces pile up across buffers; the annotation mode
    // decides whether markers pin per-batch (accumulate) or stay global and replace
    // (replace) -- the latter keeps a single running set of global PoI markers over
    // the accumulated traces.
    {
        const auto make_frame = [](std::int64_t sample) {
            auto trace_payload = std::make_shared<leakflow::base::TorchTensorPayload>(
                torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3}));
            leakflow::Buffer trace_buffer(trace_payload->caps());
            trace_buffer.set_payload(trace_payload);

            auto ann_payload = std::make_shared<leakflow::base::PlotAnnotationPayload>(
                std::vector<leakflow::base::PlotAnnotation>{
                    leakflow::base::PlotAnnotation{.sample_index = sample, .kind = "poi"},
                });
            leakflow::Buffer ann_buffer(ann_payload->caps());
            ann_buffer.set_payload(ann_payload);

            leakflow::ElementInputs inputs;
            inputs.emplace("sink", trace_buffer);
            inputs.emplace("annotations", ann_buffer);
            return inputs;
        };

        // accumulate traces + replace annotations: after two buffers the traces have
        // piled up (4 rows) but the annotation set is the latest only (1), kept global.
        auto replace_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot replace_plot(replace_runtime, "acc_replace");
        replace_plot.set_property("update_mode", std::string("accumulate"));
        replace_plot.set_property("annotation_update_mode", std::string("replace"));
        (void)replace_plot.process_inputs(make_frame(1));
        (void)replace_plot.process_inputs(make_frame(2));
        const auto& replace_snapshot = replace_runtime->trace_view()->trace_snapshots().front();
        if (!expect(replace_snapshot.trace_count() == 4, "accumulate should pile up traces to 4 rows")) {
            return 1;
        }
        if (!expect(replace_snapshot.annotations.size() == 1,
                    "annotation_update_mode=replace should keep only the latest marker set")) {
            return 1;
        }
        if (!expect(replace_snapshot.annotations.front().sample_index == 2,
                    "annotation_update_mode=replace should keep the latest buffer's marker")) {
            return 1;
        }
        if (!expect(replace_snapshot.annotations.front().trace_row == -1,
                    "replace-mode annotations should stay global (trace_row -1)")) {
            return 1;
        }

        // accumulate traces + accumulate annotations: both pile up (2 markers), and
        // each global marker pins to its fold's LAST row (batch end -- the trace the
        // slider follows while streaming). Frames are [2,3]: fold 0 rows 0..1 (last 1),
        // fold 1 rows 2..3 (last 3).
        auto accumulate_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
        plot_plugin::TracePlot accumulate_plot(accumulate_runtime, "acc_acc");
        accumulate_plot.set_property("update_mode", std::string("accumulate"));
        accumulate_plot.set_property("annotation_update_mode", std::string("accumulate"));
        (void)accumulate_plot.process_inputs(make_frame(1));
        (void)accumulate_plot.process_inputs(make_frame(2));
        const auto& accumulate_snapshot = accumulate_runtime->trace_view()->trace_snapshots().front();
        if (!expect(accumulate_snapshot.annotations.size() == 2,
                    "annotation_update_mode=accumulate should keep every marker set")) {
            return 1;
        }
        if (!expect(accumulate_snapshot.annotations.front().trace_row == 1,
                    "accumulate should pin fold 0's global marker to its last row (1)")) {
            return 1;
        }
        if (!expect(accumulate_snapshot.annotations.back().trace_row == 3,
                    "accumulate should pin fold 1's global marker to its last row (3)")) {
            return 1;
        }
    }

    return 0;
}
