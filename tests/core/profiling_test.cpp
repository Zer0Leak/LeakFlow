#include "leakflow/core/caps.hpp"
#include "leakflow/core/descriptor.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/core/pad.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/telemetry.hpp"
#include "leakflow/core/telemetry_trace.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

namespace {

bool expect(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

// A trivial source that declares one internal op scope ("work").
class ProbeSource final : public leakflow::Element {
public:
    ProbeSource() : Element("probe") {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "ProbeSource";
        descriptor.output_pads = {leakflow::Pad("src", leakflow::PadDirection::Output, leakflow::Caps("sca/test"))};
        descriptor.telemetry_specs = {leakflow::make_duration_telemetry_spec("work", "probe work")};
        configure_from_descriptor(leakflow::with_common_element_properties(descriptor));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override {
        auto scope = profile_scope("work");
        return leakflow::Buffer(leakflow::Caps("sca/test"));
    }
};

// A trivial sink so the source's output pad is linked and fired by the executor.
class ProbeSink final : public leakflow::Element {
public:
    ProbeSink() : Element("sink") {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "ProbeSink";
        descriptor.input_pads = {leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps("sca/test"))};
        descriptor.provenance_slots = 0;
        configure_from_descriptor(leakflow::with_common_element_properties(descriptor));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override { return std::nullopt; }
};

bool test_duration_stat() {
    using namespace std::chrono_literals;
    leakflow::RuntimeTelemetryDurationStat stat;
    stat.observe(100ns);
    stat.observe(300ns);
    bool ok = expect(stat.count() == 2, "duration stat count");
    ok = expect(stat.total_ns() == 400, "duration stat total") && ok;
    ok = expect(stat.min_ns() == 100, "duration stat min") && ok;
    ok = expect(stat.max_ns() == 300, "duration stat max") && ok;

    const auto report = stat.report("work");
    ok = expect(report.average_ns() == 200, "duration report average") && ok;

    // A disabled switch makes observe a no-op.
    leakflow::RuntimeTelemetrySwitch off;
    off.set_enabled(false);
    leakflow::RuntimeTelemetryDurationStat gated;
    gated.bind(off);
    gated.observe(500ns);
    ok = expect(gated.count() == 0, "disabled duration stat must not record") && ok;
    return ok;
}

bool test_scoped_timer_and_trace() {
    leakflow::TelemetryTraceSink sink;
    leakflow::RuntimeTelemetryDurationStat stat;
    {
        leakflow::RuntimeTelemetryScopedTimer timer(&stat, &sink, "a.process", "a");
    }
    bool ok = expect(stat.count() == 1, "scoped timer recorded one sample");
    ok = expect(sink.span_count() == 1, "trace sink recorded one span") && ok;
    const auto json = sink.to_chrome_json();
    ok = expect(json.find("\"name\":\"a.process\"") != std::string::npos, "trace json carries span name") && ok;
    ok = expect(json.find("\"ph\":\"X\"") != std::string::npos, "trace json is complete-event format") && ok;

    // An inactive timer (no stat, no sink) is a safe no-op.
    {
        leakflow::RuntimeTelemetryScopedTimer inactive(nullptr, nullptr, "x", "y");
    }
    ok = expect(sink.span_count() == 1, "inactive timer recorded nothing") && ok;
    return ok;
}

bool test_element_profiling_gate() {
    auto probe = std::make_shared<ProbeSource>();

    // Profiling off (default): op scope is a no-op, no reports.
    (void)probe->process(std::nullopt);
    bool ok = expect(probe->duration_reports().empty(), "no reports when profiling disabled");

    // Enable through the pipeline and run once over a linked source -> sink graph.
    auto consumer = std::make_shared<ProbeSink>();
    leakflow::Pipeline pipeline;
    leakflow::TelemetryTraceSink sink;
    pipeline.set_profiling_enabled(true);
    pipeline.set_trace_sink(&sink);
    pipeline.add(probe);
    pipeline.add(consumer);
    pipeline.link(probe, "src", consumer, "in");
    (void)pipeline.run();

    ok = expect(probe->profiling_enabled(), "pipeline mirrored profiling to element") && ok;

    const auto reports = probe->duration_reports();
    bool has_process = false;
    bool has_work = false;
    for (const auto &report : reports) {
        has_process = has_process || report.name == "process";
        has_work = has_work || report.name == "work";
        ok = expect(report.count >= 1, "recorded channel has at least one sample") && ok;
    }
    ok = expect(has_process, "executor recorded the built-in process timing") && ok;
    ok = expect(has_work, "element op scope recorded work timing") && ok;
    ok = expect(!sink.empty(), "trace sink captured spans during the run") && ok;

    // Live overlay data path: telemetry_snapshot() surfaces recorded duration
    // channels as ms with kind Duration so the --graph panel can render them.
    bool process_in_snapshot = false;
    for (const auto &snap : probe->telemetry_snapshot()) {
        if (snap.name == "process") {
            process_in_snapshot = snap.kind == leakflow::TelemetryKind::Duration && snap.unit == "ms";
        }
    }
    ok = expect(process_in_snapshot, "telemetry_snapshot surfaces process timing as a Duration channel") && ok;
    return ok;
}

} // namespace

int main() {
    bool ok = test_duration_stat();
    ok = test_scoped_timer_and_trace() && ok;
    ok = test_element_profiling_gate() && ok;
    return ok ? 0 : 1;
}
