#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/summary_document.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

bool expect(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

class RecordingObserver final : public leakflow::PipelineObserver {
public:
    void observe(const leakflow::PipelineEvent &event) override { events.push_back(event); }

    std::vector<leakflow::PipelineEvent> events;
};

class TestPayload final : public leakflow::Payload {
public:
    [[nodiscard]] std::string type_name() const override { return "test/payload"; }
    [[nodiscard]] std::string layout() const override { return "unit/channel/field"; }

    void describe(leakflow::SummarySection &section, std::int64_t) const override {
        section.add_field("payload", type_name(), leakflow::SummaryValueRole::TypeName);
        section.add_field("result", "(byte_index: 3, shape: [2, 1, 2])", leakflow::SummaryValueRole::Size);
    }
};

class SourceElement final : public leakflow::Element {
public:
    SourceElement() : Element("source") {
        set_element_identity("TestSrc", "Source/Test");
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/test")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override {
        if (input) {
            return input;
        }

        leakflow::Buffer buffer(leakflow::Caps("sca/test", {{"dtype", "float32"}}));
        buffer.set_metadata("source", "generated");
        buffer.set_payload(std::make_shared<TestPayload>());
        return buffer;
    }
};

class TransformElement final : public leakflow::Element {
public:
    TransformElement() : Element("transform") {
        set_element_identity("TestTransform", "Transform/Test");
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps("sca/test")));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/test")));
        add_property(leakflow::PropertySpec(
            "gain",
            std::int64_t{1},
            "test gain",
            "",
            std::monostate{},
            "",
            leakflow::PropertyEffect{
                .kind = leakflow::PropertyEffectKind::PayloadOutput,
                .scope = leakflow::PropertyInvalidationScope::Downstream,
                .output_pads = {"out"},
            }));
        add_telemetry(leakflow::TelemetrySpec("processed", std::int64_t{0}, "processed buffers", "buffers"));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override {
        // Exercise the progress channel: the pipeline injects a sink at start_all(), so these
        // surface as ProgressReported events. A second run begins immediately after the first
        // completion; its 0.0 must pass because start_all() resets the per-run throttle state.
        if (process_count++ == 0) {
            report_progress(0.5, "halfway");
        } else {
            report_progress(0.0, "starting again");
        }
        report_progress(1.0, "done");
        if (input) {
            input->set_metadata("transform", "visited");
        }
        return input;
    }

private:
    std::size_t process_count = 0;
};

class SinkElement final : public leakflow::Element {
public:
    SinkElement() : Element("sink") {
        set_element_identity("TestSink", "Sink/Test");
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps("sca/test")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override {
        received = input.has_value();
        return std::nullopt;
    }

    bool received = false;
};

class ProgressProbeElement final : public leakflow::Element {
public:
    explicit ProgressProbeElement(std::string name) : Element(std::move(name)) {
        set_element_identity("ProgressProbe", "Source/Test");
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override {
        return std::nullopt;
    }

    void publish(double fraction, std::string message, std::uint64_t index, std::uint64_t total,
                 leakflow::ProgressStatus status = leakflow::ProgressStatus::Active) {
        report_progress(fraction, std::move(message), index, total, status);
    }
};

} // namespace

int main() {
    leakflow::Pipeline pipeline;
    auto source = pipeline.add(std::make_shared<SourceElement>());
    auto transform = pipeline.add(std::make_shared<TransformElement>());
    auto sink = pipeline.add(std::make_shared<SinkElement>());
    transform->set_property("gain", std::int64_t{7});

    pipeline.link(source, "out", transform, "in");
    pipeline.link(transform, "out", sink, "in");
    pipeline.add_output_metadata_annotation(source, "out", {{"branch", "annotated"}});

    const auto topology = pipeline.topology_snapshot();
    if (!expect(topology.elements.size() == 3, "topology did not include all elements")) {
        return 1;
    }
    if (!expect(topology.links.size() == 2, "topology did not include all links")) {
        return 1;
    }
    if (!expect(topology.links.front().id == "source.out -> transform.in", "topology link id was unexpected")) {
        return 1;
    }

    bool saw_gain = false;
    bool saw_gain_effect = false;
    for (const auto &property : topology.elements[1].properties) {
        if (property.name == "gain" && property.value == "7" && property.value_type == "integer") {
            saw_gain = true;
            saw_gain_effect = property.effect.kind == leakflow::PropertyEffectKind::PayloadOutput &&
                              property.effect.scope == leakflow::PropertyInvalidationScope::Downstream &&
                              property.effect.output_pads == std::vector<std::string>{"out"};
        }
    }
    if (!expect(saw_gain, "topology did not capture configured element property")) {
        return 1;
    }
    if (!expect(saw_gain_effect, "topology did not capture property effect")) {
        return 1;
    }
    bool saw_telemetry = false;
    for (const auto &field : topology.elements[1].telemetry) {
        if (field.name == "processed" && field.value == "0" && field.value_type == "integer" &&
            field.unit == "buffers") {
            saw_telemetry = true;
        }
    }
    if (!expect(saw_telemetry, "topology did not capture element telemetry")) {
        return 1;
    }

    auto observer = std::make_shared<RecordingObserver>();
    pipeline.set_observer(observer);
    (void)pipeline.run();
    (void)pipeline.run();

    bool saw_topology = false;
    bool saw_started = false;
    bool saw_stopped = false;
    std::map<std::string, leakflow::PipelineBufferObservation> buffers;

    for (const auto &event : observer->events) {
        if (event.sequence == 0) {
            std::cerr << "observer event sequence was not assigned\n";
            return 1;
        }
        if (event.kind == leakflow::PipelineEventKind::TopologySnapshot && event.topology) {
            saw_topology = true;
        }
        if (event.kind == leakflow::PipelineEventKind::Started) {
            saw_started = true;
        }
        if (event.kind == leakflow::PipelineEventKind::Stopped) {
            saw_stopped = true;
        }
        if (event.kind == leakflow::PipelineEventKind::BufferObserved && event.buffer) {
            buffers.emplace(event.buffer->link_id, *event.buffer);
        }
    }

    // Progress reports pushed from process() reach the observer as ProgressReported events,
    // tagged with the reporting element (Pipeline injects the sink at start_all()).
    bool saw_progress_half = false;
    bool saw_progress_done = false;
    bool saw_second_run_reset = false;
    bool saw_first_completion = false;
    for (const auto &event : observer->events) {
        if (event.kind != leakflow::PipelineEventKind::ProgressReported || !event.progress) {
            continue;
        }
        if (!expect(event.progress->element.element_name == "transform", "progress event from wrong element")) {
            return 1;
        }
        if (event.progress->fraction == 0.5 && event.progress->message == "halfway"
            && event.progress->status == leakflow::ProgressStatus::Active) {
            saw_progress_half = true;
        }
        if (event.progress->fraction == 1.0 && event.progress->message == "done"
            && event.progress->status == leakflow::ProgressStatus::Completed) {
            saw_progress_done = true;
            saw_first_completion = true;
        }
        if (saw_first_completion && event.progress->fraction == 0.0 &&
            event.progress->message == "starting again") {
            saw_second_run_reset = true;
        }
    }
    if (!expect(saw_progress_half && saw_progress_done, "observer did not receive progress reports")) {
        return 1;
    }
    if (!expect(saw_second_run_reset,
                "new run did not reset progress to 0.0 immediately after completion")) {
        return 1;
    }

    if (!expect(saw_topology, "observer did not receive topology snapshot")) {
        return 1;
    }
    if (!expect(saw_started, "observer did not receive started event")) {
        return 1;
    }
    if (!expect(saw_stopped, "observer did not receive stopped event")) {
        return 1;
    }
    if (!expect(buffers.size() == 2, "observer did not receive routed buffer observations")) {
        return 1;
    }
    if (!expect(buffers.at("source.out -> transform.in").buffer.metadata.at("branch") == "annotated",
                "observer buffer did not include output metadata annotation")) {
        return 1;
    }
    if (!expect(buffers.at("transform.out -> sink.in").buffer.metadata.at("transform") == "visited",
                "observer buffer did not include transform metadata")) {
        return 1;
    }
    if (!expect(buffers.at("transform.out -> sink.in").buffer.has_payload,
                "observer buffer did not report payload presence")) {
        return 1;
    }
    if (!expect(buffers.at("transform.out -> sink.in").buffer.payload_type == "test/payload",
                "observer buffer did not report payload type")) {
        return 1;
    }
    if (!expect(!buffers.at("transform.out -> sink.in").buffer.payload_summary.empty(),
                "observer buffer did not report payload summary")) {
        return 1;
    }
    if (!expect(buffers.at("transform.out -> sink.in").buffer.payload_summary.front()
                    == "result=(byte_index: 3, shape: [2, 1, 2])",
                "observer buffer payload summary was wrong")) {
        return 1;
    }

    // A requested Stop closes only progress entries that are still Active. Explicit terminal
    // reports remain unchanged, cancellation precedes Pipeline Stopped, and the detached sink
    // suppresses any report attempted after teardown.
    leakflow::Pipeline stop_pipeline;
    auto active = std::make_shared<ProgressProbeElement>("active");
    auto completed = std::make_shared<ProgressProbeElement>("completed");
    auto explicitly_cancelled = std::make_shared<ProgressProbeElement>("explicitly_cancelled");
    stop_pipeline.add(active);
    stop_pipeline.add(completed);
    stop_pipeline.add(explicitly_cancelled);
    auto stop_observer = std::make_shared<RecordingObserver>();
    stop_pipeline.set_observer(stop_observer);
    std::stop_source stop_source;
    stop_pipeline.set_stop_token(stop_source.get_token());
    stop_pipeline.start_all();
    active->publish(0.4, "working", 4, 10);
    completed->publish(1.0, "complete", 10, 10, leakflow::ProgressStatus::Completed);
    explicitly_cancelled->publish(0.6, "explicit cancel", 6, 10, leakflow::ProgressStatus::Cancelled);
    stop_source.request_stop();
    stop_pipeline.stop_all();
    const auto event_count_after_stop = stop_observer->events.size();
    // A terminal report bypasses Element's 33 ms Active-report throttle, so this specifically
    // proves teardown detached/closed the sink rather than accidentally coalescing the call.
    active->publish(1.0, "late completion", 10, 10, leakflow::ProgressStatus::Completed);

    std::uint64_t stopped_sequence = 0;
    int active_cancelled_count = 0;
    int completed_count = 0;
    int completed_cancelled_count = 0;
    int explicit_cancelled_count = 0;
    for (const auto &event : stop_observer->events) {
        if (event.kind == leakflow::PipelineEventKind::Stopped) {
            stopped_sequence = event.sequence;
            continue;
        }
        if (event.kind != leakflow::PipelineEventKind::ProgressReported || !event.progress) {
            continue;
        }
        if (!expect(stopped_sequence == 0, "progress event arrived after Pipeline Stopped")) {
            return 1;
        }
        if (event.progress->element.element_name == "active"
            && event.progress->status == leakflow::ProgressStatus::Cancelled) {
            ++active_cancelled_count;
            if (!expect(event.progress->fraction == 1.0 && event.progress->message == "cancelled"
                            && event.progress->index == 4 && event.progress->total == 10,
                        "synthesized cancellation did not preserve active progress counters")) {
                return 1;
            }
        }
        if (event.progress->element.element_name == "completed") {
            if (event.progress->status == leakflow::ProgressStatus::Completed) {
                ++completed_count;
            } else if (event.progress->status == leakflow::ProgressStatus::Cancelled) {
                ++completed_cancelled_count;
            }
        }
        if (event.progress->element.element_name == "explicitly_cancelled"
            && event.progress->status == leakflow::ProgressStatus::Cancelled) {
            ++explicit_cancelled_count;
        }
    }
    if (!expect(stopped_sequence != 0 && active_cancelled_count == 1,
                "requested Stop did not emit exactly one cancellation for active progress")) {
        return 1;
    }
    if (!expect(completed_count == 1 && completed_cancelled_count == 0,
                "requested Stop replaced or duplicated explicit completion")) {
        return 1;
    }
    if (!expect(explicit_cancelled_count == 1,
                "requested Stop duplicated explicit cancellation")) {
        return 1;
    }
    if (!expect(stop_observer->events.size() == event_count_after_stop,
                "detached progress sink forwarded a report after Pipeline Stopped")) {
        return 1;
    }

    return 0;
}
