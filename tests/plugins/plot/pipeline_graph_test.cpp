#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/plot/pipeline_graph.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

leakflow::PipelineBufferObservation observation(std::string value, std::vector<std::uint32_t> provenance = {}) {
    leakflow::PipelineBufferObservation observed;
    observed.link_id = "source.out -> sink.in";
    observed.source.element_name = "source";
    observed.source.pad_name = "out";
    observed.sink.element_name = "sink";
    observed.sink.pad_name = "in";
    observed.buffer.caps = leakflow::Caps("sca/test");
    observed.buffer.metadata.emplace("value", std::move(value));
    observed.buffer.has_payload = true;
    observed.buffer.payload_type = "test/payload";
    observed.buffer.payload_summary.push_back("result=(byte_index: 3, shape: [2, 1, 2])");
    observed.sequence = 42;
    observed.provenance = std::move(provenance);
    return observed;
}

class ControlElement final : public leakflow::Element {
public:
    explicit ControlElement(std::string name) : Element(std::move(name)) {
        set_element_identity("ControlElement", "PassThrough/Test");
        add_property(leakflow::PropertySpec("enabled", true, "enable the test control"));
        add_property(leakflow::PropertySpec("count", std::int64_t{1}, "test count", "",
                                            leakflow::IntRangeConstraint{0, 10}, "",
                                            leakflow::PropertyEffect{
                                                .kind = leakflow::PropertyEffectKind::PayloadOutput,
                                                .scope = leakflow::PropertyInvalidationScope::Downstream,
                                                .output_pads = {"out"},
                                            }));
        add_property(leakflow::PropertySpec("mode", std::string("a"), "test mode", "",
                                            leakflow::StringEnumConstraint{{"a", "b"}}));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override { return input; }
};

} // namespace

int main() {
    leakflow::plot::PipelineGraphRuntime runtime;

    leakflow::PipelineTopologySnapshot topology;
    topology.elements.push_back(leakflow::PipelineElementSnapshot{
        .type_name = "TestSrc",
        .name = "source",
        .klass = "Source/Test",
    });
    topology.elements.push_back(leakflow::PipelineElementSnapshot{
        .type_name = "TestSink",
        .name = "sink",
        .klass = "Sink/Test",
    });
    topology.links.push_back(leakflow::PipelineLinkSnapshot{
        .id = "source.out -> sink.in",
        .source =
            leakflow::PipelineEndpointSnapshot{
                .element_type = "TestSrc",
                .element_name = "source",
                .element_klass = "Source/Test",
                .pad_name = "out",
            },
        .sink =
            leakflow::PipelineEndpointSnapshot{
                .element_type = "TestSink",
                .element_name = "sink",
                .element_klass = "Sink/Test",
                .pad_name = "in",
            },
        .source_caps = leakflow::Caps("sca/test"),
        .sink_caps = leakflow::Caps("sca/test"),
    });

    runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::TopologySnapshot,
        .topology = topology,
    });
    runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::Started,
    });
    runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::BufferObserved,
        .buffer = observation("first", {0, 1, 1}),
    });
    runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::BufferObserved,
        .buffer = observation("latest", {0, 2, 1}),
    });

    runtime.drain_events();
    if (!expect(runtime.has_topology(), "graph runtime did not store topology")) {
        return 1;
    }
    if (!expect(runtime.running(), "graph runtime did not enter running state")) {
        return 1;
    }
    if (!expect(runtime.topology().elements.size() == 2, "graph runtime topology element count was wrong")) {
        return 1;
    }
    if (!expect(runtime.observed_count("source.out -> sink.in") == 2,
                "graph runtime did not count coalesced buffer events")) {
        return 1;
    }
    if (!expect(runtime.latest_buffers().at("source.out -> sink.in").buffer.metadata.at("value") == "latest",
                "graph runtime did not retain latest buffer observation")) {
        return 1;
    }
    if (!expect(runtime.latest_buffers().at("source.out -> sink.in").buffer.payload_summary.front()
                    == "result=(byte_index: 3, shape: [2, 1, 2])",
                "graph runtime did not retain payload summary")) {
        return 1;
    }
    // Vector-clock provenance aggregation (Phase 27): component-wise max over
    // observed buffers ({0,1,1} then {0,2,1} -> {0,2,1}).
    if (!expect(runtime.max_provenance().size() >= 3 && runtime.max_provenance()[1] == 2 &&
                    runtime.max_provenance()[2] == 1,
                "graph runtime did not aggregate max provenance")) {
        return 1;
    }
    // Pinned info panel toggle (item 1).
    runtime.toggle_pinned_element("source");
    if (!expect(runtime.is_element_pinned("source"), "graph runtime did not pin element")) {
        return 1;
    }
    runtime.toggle_pinned_element("source");
    if (!expect(!runtime.is_element_pinned("source"), "graph runtime did not unpin element")) {
        return 1;
    }
    runtime.toggle_pinned_link("source.out -> sink.in");
    if (!expect(runtime.is_link_pinned("source.out -> sink.in"), "graph runtime did not pin link")) {
        return 1;
    }

    runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::Error,
        .message = "boom",
    });
    runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::Stopped,
    });
    runtime.drain_events();

    if (!expect(!runtime.running(), "graph runtime did not leave running state after error/stop")) {
        return 1;
    }
    if (!expect(runtime.stopped(), "graph runtime did not enter stopped state")) {
        return 1;
    }
    if (!expect(runtime.last_error() && *runtime.last_error() == "boom", "graph runtime did not store latest error")) {
        return 1;
    }

    runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::CommandRejected,
        .command = leakflow::PipelineCommandObservation{
            .status = leakflow::PipelineCommandStatus::Rejected,
            .detail = "property change blocked by incremental correlation",
        },
    });
    runtime.drain_events();
    if (!expect(runtime.last_error()
                    && *runtime.last_error() == "property change blocked by incremental correlation",
            "graph runtime did not surface command rejection detail")) {
        return 1;
    }

    runtime.clear();
    if (!expect(!runtime.has_topology(), "graph runtime clear did not reset topology")) {
        return 1;
    }

    leakflow::Pipeline pipeline;
    auto control_element = pipeline.add(std::make_shared<ControlElement>("control"));
    leakflow::plot::PipelineControlRuntime controls;
    controls.bind(pipeline);

    if (!expect(controls.has_element("control"), "control runtime did not bind pipeline element")) {
        return 1;
    }
    if (!expect(controls.element_names().size() == 1 && controls.element_names().front() == "control",
                "control runtime element names were wrong")) {
        return 1;
    }

    controls.open("control");
    if (!expect(controls.is_open("control"), "control runtime did not open element controls")) {
        return 1;
    }
    if (!expect(controls.take_focus_request("control"),
                "control runtime did not request focus when opening element controls")) {
        return 1;
    }
    if (!expect(!controls.take_focus_request("control"), "control runtime focus request was not consumed")) {
        return 1;
    }
    controls.open("control");
    if (!expect(controls.take_focus_request("control"),
                "control runtime did not request focus when reopening already-open element controls")) {
        return 1;
    }
    controls.close("control");
    if (!expect(!controls.is_open("control"), "control runtime did not close element controls")) {
        return 1;
    }

    auto control_graph = std::make_shared<leakflow::plot::PipelineGraphRuntime>();
    control_graph->observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::TopologySnapshot,
        .topology = pipeline.topology_snapshot(),
    });
    control_graph->drain_events();
    controls.set_observer(control_graph);

    // Provenance slot allocation flows into the topology snapshot (Phase 27):
    // the control element claims one slot at base index 1 (0 is reserved).
    if (!expect(control_graph->topology().elements.front().provenance_slots == 1 &&
                    control_graph->topology().elements.front().provenance_base == 1,
                "topology snapshot did not carry the provenance slot allocation")) {
        return 1;
    }

    if (!expect(controls.set_property("control", "enabled", false), "control runtime did not set bool property")) {
        return 1;
    }
    if (!expect(control_element->property_as<bool>("enabled") == std::optional<bool>{false},
                "control runtime bool property did not reach element")) {
        return 1;
    }
    if (!expect(controls.set_property("control", "count", std::int64_t{7}),
                "control runtime did not set integer property")) {
        return 1;
    }
    if (!expect(control_element->property_as<std::int64_t>("count") == std::optional<std::int64_t>{7},
                "control runtime integer property did not reach element")) {
        return 1;
    }
    if (!expect(controls.set_property("control", "mode", std::string("b")),
                "control runtime did not set enum string property")) {
        return 1;
    }

    const auto changes = controls.take_changes();
    if (!expect(changes.size() == 3, "control runtime did not record property changes")) {
        return 1;
    }
    if (!expect(changes[1].element_name == "control" && changes[1].property_name == "count" &&
                    changes[1].previous_value == "1" && changes[1].new_value == "7",
                "control runtime change record was wrong")) {
        return 1;
    }
    if (!expect(changes[1].effect.kind == leakflow::PropertyEffectKind::PayloadOutput &&
                    changes[1].effect.scope == leakflow::PropertyInvalidationScope::Downstream &&
                    changes[1].effect.output_pads == std::vector<std::string>{"out"},
                "control runtime change record effect was wrong")) {
        return 1;
    }
    if (!expect(controls.take_changes().empty(), "control runtime did not clear change records")) {
        return 1;
    }

    control_graph->drain_events();
    bool saw_property_event = false;
    for (const auto &event : control_graph->recent_events()) {
        if (event.kind == leakflow::PipelineEventKind::PropertyChanged && event.property_change &&
            event.property_change->element.element_name == "control" &&
            event.property_change->property_name == "count" &&
            event.property_change->new_value == "7" &&
            event.property_change->effect.kind == leakflow::PropertyEffectKind::PayloadOutput) {
            saw_property_event = true;
        }
    }
    if (!expect(saw_property_event, "control runtime did not emit property changed observer event")) {
        return 1;
    }
    bool saw_updated_topology_value = false;
    for (const auto &property : control_graph->topology().elements.front().properties) {
        if (property.name == "count" && property.value == "7") {
            saw_updated_topology_value = true;
        }
    }
    if (!expect(saw_updated_topology_value, "graph runtime did not apply property changed event to topology")) {
        return 1;
    }

    leakflow::plot::PipelineGraphRuntime readonly_runtime;
    leakflow::PipelineTopologySnapshot readonly_topology;
    readonly_topology.elements.push_back(leakflow::PipelineElementSnapshot{
        .type_name = "Queue",
        .name = "q",
        .klass = "PassThrough/Flow/Queue",
        .telemetry = {
            leakflow::PipelineTelemetrySnapshot{
                .name = "size",
                .value_type = "integer",
                .value = "0",
                .unit = "buffers",
            },
        },
    });
    readonly_runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::TopologySnapshot,
        .topology = readonly_topology,
    });
    readonly_runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::TelemetryChanged,
        .telemetry_change =
            leakflow::PipelineTelemetryChangeObservation{
                .element =
                    leakflow::PipelineEndpointSnapshot{
                        .element_type = "Queue",
                        .element_name = "q",
                        .element_klass = "PassThrough/Flow/Queue",
                },
                .telemetry_name = "size",
                .value_type = "integer",
                .previous_value = "0",
                .new_value = "3",
                .unit = "buffers",
                .description = "current number of buffers held by this queue",
            },
    });
    readonly_runtime.drain_events();
    bool saw_telemetry_update = false;
    bool kept_telemetry_separate = readonly_runtime.topology().elements.front().properties.empty();
    for (const auto &field : readonly_runtime.topology().elements.front().telemetry) {
        if (field.name == "size") {
            saw_telemetry_update = field.value == "3" && field.unit == "buffers" &&
                                   field.description == "current number of buffers held by this queue";
        }
    }
    if (!expect(saw_telemetry_update && kept_telemetry_separate,
                "graph runtime did not apply telemetry update to topology telemetry")) {
        return 1;
    }

    leakflow::plot::PipelineGraphRuntime disabled_telemetry_runtime;
    auto disabled_telemetry_topology = readonly_topology;
    disabled_telemetry_topology.runtime_telemetry_enabled = false;
    disabled_telemetry_runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::TopologySnapshot,
        .topology = disabled_telemetry_topology,
    });
    disabled_telemetry_runtime.observe(leakflow::PipelineEvent{
        .kind = leakflow::PipelineEventKind::TelemetryChanged,
        .telemetry_change =
            leakflow::PipelineTelemetryChangeObservation{
                .element =
                    leakflow::PipelineEndpointSnapshot{
                        .element_type = "Queue",
                        .element_name = "q",
                        .element_klass = "PassThrough/Flow/Queue",
                },
                .telemetry_name = "size",
                .value_type = "integer",
                .previous_value = "0",
                .new_value = "9",
                .unit = "buffers",
            },
    });
    disabled_telemetry_runtime.drain_events();
    const auto& disabled_telemetry_topology_after = disabled_telemetry_runtime.topology();
    if (!expect(!disabled_telemetry_topology_after.runtime_telemetry_enabled &&
                    disabled_telemetry_topology_after.elements.front().telemetry.front().value == "0",
            "graph runtime did not keep disabled telemetry visually steady")) {
        return 1;
    }

    if (!expect(!controls.set_property("control", "count", std::int64_t{99}),
                "control runtime accepted invalid constrained property value")) {
        return 1;
    }
    if (!expect(controls.last_error().has_value(), "control runtime did not retain validation error")) {
        return 1;
    }

    controls.set_edits_enabled(false);
    if (!expect(!controls.set_property("control", "enabled", true),
                "control runtime accepted edits while disabled")) {
        return 1;
    }

    return 0;
}
