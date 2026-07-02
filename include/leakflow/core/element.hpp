#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/descriptor.hpp"
#include "leakflow/core/pad.hpp"
#include "leakflow/core/property.hpp"
#include "leakflow/core/telemetry_trace.hpp"
#include "leakflow/log/logger.hpp"

#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

using ElementInputs = std::map<std::string, std::optional<Buffer>>;

// Per-pad output contract (Phase 27): a buffer per produced output pad. The
// mirror of ElementInputs. See docs/design/dataflow_sync_model.md.
using ElementOutputs = std::map<std::string, Buffer>;

class Element {
public:
    explicit Element(std::string name);
    virtual ~Element() = default;

    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] bool name_locked() const;
    [[nodiscard]] const std::string& element_type() const;
    [[nodiscard]] const std::string& element_kclass() const;
    // Vector-clock provenance slots this element claims (Phase 27); see
    // ElementDescriptor::provenance_slots.
    [[nodiscard]] int provenance_slots() const;

    // Live-source flag (live phase): true for a streaming/capture source.
    // Detected by run() to choose the pump loop; default false (one-run).
    [[nodiscard]] bool is_live() const;

    // Propagated graph liveness: true when this element is a live source or has
    // any live source upstream. Pipeline refreshes this when topology changes
    // and before start(), so stateful elements can select live-aware behavior.
    [[nodiscard]] bool is_live_driven() const;

    // Thread-boundary flag (live phase): true for a Queue, which cuts the pipeline
    // into segments handed off across threads. Segment decomposition splits here.
    // Default false. See docs/design/dataflow_sync_model.md S10.
    [[nodiscard]] bool is_thread_boundary() const;

    // Streaming end-of-stream signal (live phase): a streaming source returns true
    // once it has emitted its whole stream. The run() pump stops when every live
    // source is at end-of-stream. Default false (one-run/non-streaming elements).
    [[nodiscard]] virtual bool at_end_of_stream() const;
    [[nodiscard]] const std::vector<Pad>& pad_templates() const;
    [[nodiscard]] const std::vector<Pad>& input_pads() const;
    [[nodiscard]] const std::vector<Pad>& output_pads() const;

    void add_pad_template(Pad pad);
    void add_input_pad(Pad pad);
    void add_output_pad(Pad pad);
    [[nodiscard]] bool can_request_input_pad(std::string_view name) const;
    [[nodiscard]] bool can_request_output_pad(std::string_view name) const;
    [[nodiscard]] bool can_request_pad(std::string_view name) const;
    bool request_input_pad(std::string_view name);
    bool request_output_pad(std::string_view name);

    void add_property(PropertySpec spec);
    void add_telemetry(TelemetrySpec spec);
    void configure_from_descriptor(const ElementDescriptor& descriptor);

    [[nodiscard]] const std::vector<PropertySpec>& property_specs() const;
    [[nodiscard]] const std::map<std::string, PropertyValue>& properties() const;
    [[nodiscard]] const std::vector<TelemetrySpec>& telemetry_specs() const;
    [[nodiscard]] virtual std::vector<TelemetrySnapshot> telemetry_snapshot() const;
    void set_runtime_telemetry_enabled(bool enabled);
    [[nodiscard]] bool runtime_telemetry_enabled() const;

    // Profiling (timing) is gated separately from size telemetry: the clock read
    // has a cost and a self-observer effect, so it is opt-in (--print-profile /
    // --profile-file). See docs/design/profiling.md.
    void set_profiling_enabled(bool enabled);
    [[nodiscard]] bool profiling_enabled() const;
    // Trace sink for per-event Chrome-trace spans. Null = aggregate-only profiling.
    // Mirrored from the pipeline; not owned.
    void set_trace_sink(TelemetryTraceSink* sink);

    // Aggregate timing of every duration channel this element accumulated in the
    // current run (the built-in "process" timer plus any declared op scopes).
    // Read after the run (or after threads join); the table renders these rows.
    [[nodiscard]] std::vector<TelemetryDurationReport> duration_reports() const;
    [[nodiscard]] bool has_property(std::string_view name) const;
    [[nodiscard]] const PropertyValue& property(std::string_view name) const;

    void validate_property_change(std::string_view name, const PropertyValue& value) const;
    void set_property(std::string_view name, PropertyValue value);

    template <typename T>
    [[nodiscard]] std::optional<T> property_as(std::string_view name) const
    {
        const auto found = properties_.find(std::string(name));
        if (found == properties_.end()) {
            return std::nullopt;
        }

        return property_value_as<T>(found->second);
    }

    virtual void start();
    virtual std::optional<Buffer> process(std::optional<Buffer> input) = 0;
    virtual std::optional<Buffer> process_inputs(ElementInputs inputs);

    // Per-pad output entry point used by the executor (Phase 27). The default
    // routes the single process_inputs() result to the element's sole output pad
    // (empty key when the element has no output pad). Multi-output elements such
    // as Tee override this to emit a distinct buffer per output pad; broadcast is
    // a Tee behavior, not an engine rule.
    virtual ElementOutputs process_pads(ElementInputs inputs);

    virtual void stop();

    // Cooperative stop (live phase, S11.8). The pipeline forwards a stop token to
    // every element before pumping; an element with a blocking wait (e.g. a paced
    // live source) polls stop_token() so Ctrl+C / window-close interrupts the wait
    // promptly instead of running it to completion. Elements that never block can
    // ignore it. Default token has no stop-state, so stop_requested() stays false.
    void set_stop_token(std::stop_token token);

    // Replay-safety signal (Phase 25). Returns true when process_inputs() is a
    // deterministic function of (inputs, current property values) between
    // start() and stop(), so the session may replay it during a partial rerun.
    // Stateful elements (e.g. Queue) override to false, forcing escalation to a
    // full restart when they fall inside a replay-set. See
    // docs/design/pipeline_controller.md.
    [[nodiscard]] virtual bool can_replay() const;

    [[nodiscard]] log::LogRecord make_log_record(
        log::LogLevel level,
        std::string component,
        std::string message) const;

protected:
    void set_element_identity(std::string type_name, std::string kclass);
    void set_read_only_property(std::string_view name, PropertyValue value);
    virtual void property_changed(std::string_view name);
    virtual void live_driven_changed();

    // The cooperative-stop token for this element (see set_stop_token). Subclasses
    // with blocking waits poll it; the default token never reports a stop request.
    [[nodiscard]] const std::stop_token& stop_token() const;
    [[nodiscard]] RuntimeTelemetrySwitch& runtime_telemetry();
    [[nodiscard]] const RuntimeTelemetrySwitch& runtime_telemetry() const;

    // Time an internal operation. Returns an RAII timer that records into the named
    // duration channel (which must be declared as a duration TelemetrySpec) and,
    // when a trace sink is set, emits a span. A no-op when profiling is disabled or
    // the channel was not declared, so calling it unconditionally in process() is
    // cheap. Example: auto scope = profile_scope("hamming_weight");
    [[nodiscard]] RuntimeTelemetryScopedTimer profile_scope(std::string_view name);

private:
    friend class Pipeline;

    void lock_name();
    void set_live_driven(bool live_driven);

    // Lazily-bound duration accumulator for a declared duration channel; null when
    // the channel was not declared. process_stat() is the built-in "process" timer
    // the executor feeds.
    [[nodiscard]] RuntimeTelemetryDurationStat* duration_stat(std::string_view name);
    [[nodiscard]] RuntimeTelemetryDurationStat* process_stat();
    [[nodiscard]] TelemetryTraceSink* trace_sink() const;
    RuntimeTelemetryDurationStat& ensure_duration_stat(const std::string& name);

    std::string name_;
    bool name_locked_ = false;
    std::string element_type_;
    std::string element_kclass_;
    std::vector<Pad> pad_templates_;
    std::vector<Pad> input_pads_;
    std::vector<Pad> output_pads_;
    std::vector<PropertySpec> property_specs_;
    std::vector<TelemetrySpec> telemetry_specs_;
    std::map<std::string, PropertyValue> properties_;
    int provenance_slots_ = 1;
    bool live_source_ = false;
    bool live_driven_ = false;
    bool thread_boundary_ = false;
    RuntimeTelemetrySwitch runtime_telemetry_;
    RuntimeTelemetrySwitch profiling_;
    TelemetryTraceSink* trace_sink_ = nullptr;
    std::map<std::string, std::unique_ptr<RuntimeTelemetryDurationStat>> duration_stats_;
    RuntimeTelemetryDurationStat* process_stat_ = nullptr;
    std::stop_token stop_token_;
};

} // namespace leakflow
