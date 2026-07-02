#include "leakflow/core/pipeline_session.hpp"

#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_observer.hpp"
#include "leakflow/core/provenance.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
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

using leakflow::Buffer;
using leakflow::Caps;
using leakflow::Element;
using leakflow::Pad;
using leakflow::PadDirection;
using leakflow::PropertyEffect;
using leakflow::PropertyEffectKind;
using leakflow::PropertyInvalidationScope;
using leakflow::PropertySpec;

PropertyEffect payload_effect()
{
    return PropertyEffect{PropertyEffectKind::PayloadOutput, PropertyInvalidationScope::Downstream, {"src"}};
}

PropertyEffect ui_effect()
{
    return PropertyEffect{PropertyEffectKind::UiControl, PropertyInvalidationScope::None, {}};
}

// Source whose output payload (carried here as the "value" metadata) is a pure
// function of its "value" property. "value" is payload-output; "label" is
// ui-control.
class NumberSource final : public Element {
public:
    explicit NumberSource(std::string name)
        : Element(std::move(name))
    {
        add_output_pad(Pad("src", PadDirection::Output, Caps("test/buf")));
        add_property(PropertySpec("value", std::int64_t{0}, "", "", std::monostate{}, "", payload_effect()));
        add_property(PropertySpec("label", std::string("a"), "", "", std::monostate{}, "", ui_effect()));
    }

    std::optional<Buffer> process(std::optional<Buffer> /*input*/) override
    {
        ++process_count;
        Buffer buffer(Caps("test/buf"));
        buffer.set_metadata("value", std::to_string(property_as<std::int64_t>("value").value_or(0)));
        return buffer;
    }

    int process_count = 0;
};

// Like NumberSource, but declares itself a live source (S11.5). A property change
// on a live-driven element must apply forward, not re-emit from cache.
class LiveNumberSource final : public Element {
public:
    explicit LiveNumberSource(std::string name)
        : Element(std::move(name))
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "LiveNumberSource";
        descriptor.klass = "Source/Live/Test";
        descriptor.output_pads = {Pad("src", PadDirection::Output, Caps("test/buf"))};
        descriptor.property_specs = {
            PropertySpec("value", std::int64_t{0}, "", "", std::monostate{}, "", payload_effect()),
            PropertySpec("label", std::string("a"), "", "", std::monostate{}, "", ui_effect()),
        };
        descriptor.live_source = true;
        configure_from_descriptor(descriptor);
    }

    std::optional<Buffer> process(std::optional<Buffer> /*input*/) override
    {
        ++process_count;
        Buffer buffer(Caps("test/buf"));
        buffer.set_metadata("value", std::to_string(property_as<std::int64_t>("value").value_or(0)));
        return buffer;
    }

    int process_count = 0;
};

class PassThrough final : public Element {
public:
    explicit PassThrough(std::string name)
        : Element(std::move(name))
    {
        add_input_pad(Pad("sink", PadDirection::Input, Caps("test/buf")));
        add_output_pad(Pad("src", PadDirection::Output, Caps("test/buf")));
    }

    std::optional<Buffer> process(std::optional<Buffer> input) override
    {
        ++process_count;
        return input;
    }

    int process_count = 0;
};

class CapturingSink final : public Element {
public:
    CapturingSink(std::string name, bool replay)
        : Element(std::move(name))
        , replay_(replay)
    {
        add_input_pad(Pad("sink", PadDirection::Input, Caps("test/buf")));
    }

    void start() override { ++start_count; }

    std::optional<Buffer> process(std::optional<Buffer> input) override
    {
        ++process_count;
        if (input) {
            last_value = input->metadata_or("value", "");
            last_generation = leakflow::provenance_generation(input->provenance());
        }
        return std::nullopt;
    }

    [[nodiscard]] bool can_replay() const override { return replay_; }

    int start_count = 0;
    int process_count = 0;
    std::string last_value;
    std::uint32_t last_generation = 0;

private:
    bool replay_;
};

class RecordingObserver final : public leakflow::PipelineObserver {
public:
    void observe(const leakflow::PipelineEvent& event) override
    {
        using leakflow::PipelineCommandStatus;
        using leakflow::PipelineEventKind;
        switch (event.kind) {
        case PipelineEventKind::CommandAccepted:
            ++accepted;
            break;
        case PipelineEventKind::CommandRejected:
            ++rejected;
            if (event.command) {
                last_rejection_detail = event.command->detail;
            }
            break;
        case PipelineEventKind::CommandApplied:
            if (event.command && event.command->status == PipelineCommandStatus::Failed) {
                ++failed;
            } else {
                ++applied;
            }
            break;
        case PipelineEventKind::BufferObserved:
            if (event.buffer) {
                last_buffer_generation = event.buffer->generation;
            }
            break;
        case PipelineEventKind::ElementStarted:
            ++element_started;
            break;
        default:
            break;
        }
    }

    int accepted = 0;
    int rejected = 0;
    int applied = 0;
    int failed = 0;
    int element_started = 0;
    std::uint32_t last_buffer_generation = 0;
    std::string last_rejection_detail;
};

leakflow::Pipeline make_chain(std::shared_ptr<NumberSource>& src, std::shared_ptr<PassThrough>& mid,
                                    std::shared_ptr<CapturingSink>& sink, bool sink_replay)
{
    leakflow::Pipeline pipeline;
    src = std::make_shared<NumberSource>("src");
    mid = std::make_shared<PassThrough>("mid");
    sink = std::make_shared<CapturingSink>("sink", sink_replay);
    pipeline.add(src);
    pipeline.add(mid);
    pipeline.add(sink);
    pipeline.link(src, "src", mid, "sink");
    pipeline.link(mid, "src", sink, "sink");
    return pipeline;
}

leakflow::Pipeline make_live_chain(std::shared_ptr<LiveNumberSource>& src, std::shared_ptr<PassThrough>& mid,
                                   std::shared_ptr<CapturingSink>& sink, bool sink_replay)
{
    leakflow::Pipeline pipeline;
    src = std::make_shared<LiveNumberSource>("livesrc");
    mid = std::make_shared<PassThrough>("mid");
    sink = std::make_shared<CapturingSink>("sink", sink_replay);
    pipeline.add(src);
    pipeline.add(mid);
    pipeline.add(sink);
    pipeline.link(src, "src", mid, "sink");
    pipeline.link(mid, "src", sink, "sink");
    return pipeline;
}

} // namespace

int main()
{
    // Initial sweep: epoch 1, default value propagates downstream.
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        auto observer = std::make_shared<RecordingObserver>();
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/true));
        session.set_observer(observer);

        (void)session.run_sweep();
        if (!expect(session.generation() == 1, "initial epoch was not 1")) {
            return 1;
        }
        if (!expect(sink->last_value == "0", "initial value did not propagate")) {
            return 1;
        }
        if (!expect(sink->last_generation == 1, "initial buffer epoch was not 1")) {
            return 1;
        }
        if (!expect(sink->process_count == 1, "sink should have processed once")) {
            return 1;
        }

        // payload-output change reruns downstream and bumps the epoch.
        session.submit({"src", "value", std::int64_t{7}});
        if (!expect(session.pending_command_count() == 1, "command was not queued")) {
            return 1;
        }
        if (!expect(sink->last_value == "0", "command applied before drain")) {
            return 1;
        }
        const auto applied = session.drain_commands();
        if (!expect(applied == 1, "drain did not apply one command")) {
            return 1;
        }
        if (!expect(session.generation() == 2, "dataflow change did not bump epoch")) {
            return 1;
        }
        if (!expect(sink->last_value == "7", "downstream rerun did not propagate new value")) {
            return 1;
        }
        if (!expect(sink->last_generation == 2, "reran buffer did not carry the new epoch")) {
            return 1;
        }
        if (!expect(sink->process_count == 2, "downstream rerun did not reprocess the sink")) {
            return 1;
        }
        if (!expect(src->process_count == 2, "source-output rerun did not reprocess the source")) {
            return 1;
        }
        if (!expect(sink->start_count == 1, "replayable rerun must not restart elements")) {
            return 1;
        }
        if (!expect(observer->accepted == 1 && observer->applied == 1 && observer->rejected == 0,
                    "command events were not emitted as expected")) {
            return 1;
        }
        if (!expect(observer->last_buffer_generation == 2, "observation epoch was not updated")) {
            return 1;
        }
    }

    // ui-control change: no rerun, no epoch bump.
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/true));
        (void)session.run_sweep();
        const auto before = sink->process_count;
        session.submit({"src", "label", std::string("b")});
        (void)session.drain_commands();
        if (!expect(session.generation() == 1, "ui-control change bumped the epoch")) {
            return 1;
        }
        if (!expect(sink->process_count == before, "ui-control change triggered a rerun")) {
            return 1;
        }
    }

    // Validation: unknown element and unknown property are rejected.
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        auto observer = std::make_shared<RecordingObserver>();
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/true));
        session.set_observer(observer);
        (void)session.run_sweep();

        session.submit({"nope", "value", std::int64_t{1}});
        session.submit({"src", "nope", std::int64_t{1}});
        (void)session.drain_commands();
        if (!expect(observer->rejected == 2, "invalid commands were not rejected")) {
            return 1;
        }
        if (!expect(session.generation() == 1, "rejected commands changed the epoch")) {
            return 1;
        }
    }

    // A cached rerun crossing a non-replayable downstream element is rejected
    // transactionally instead of silently restarting or double-processing data.
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        auto observer = std::make_shared<RecordingObserver>();
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/false));
        session.set_observer(observer);
        (void)session.run_sweep();
        if (!expect(sink->start_count == 1, "sink was not started once initially")) {
            return 1;
        }
        session.submit({"src", "value", std::int64_t{3}});
        (void)session.drain_commands();
        if (!expect(sink->start_count == 1, "rejected non-replayable change restarted the pipeline")) {
            return 1;
        }
        if (!expect(src->property_as<std::int64_t>("value") == std::optional<std::int64_t>{0},
                    "rejected non-replayable change mutated the source property")) {
            return 1;
        }
        if (!expect(sink->last_value == "0", "rejected non-replayable change propagated a new value")) {
            return 1;
        }
        if (!expect(session.generation() == 1, "rejected non-replayable change bumped the generation")) {
            return 1;
        }
        if (!expect(observer->rejected == 1 && observer->accepted == 0 && observer->applied == 0,
                    "non-replayable change did not emit only a rejection")) {
            return 1;
        }
        if (!expect(observer->last_rejection_detail.find("sink") != std::string::npos,
                    "non-replayable rejection did not identify the blocking element")) {
            return 1;
        }
    }

    // Coalescing: only the last value for a (element, property) is applied.
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/true));
        (void)session.run_sweep();
        session.submit({"src", "value", std::int64_t{1}});
        session.submit({"src", "value", std::int64_t{2}});
        if (!expect(session.pending_command_count() == 1, "commands were not coalesced")) {
            return 1;
        }
        (void)session.drain_commands();
        if (!expect(sink->last_value == "2", "coalesced command did not keep the last value")) {
            return 1;
        }
    }

    // Caching off: partial rerun is disabled; dataflow change falls back to a
    // full sweep and still propagates.
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/true));
        session.set_caching_enabled(false);
        (void)session.run_sweep();
        session.submit({"src", "value", std::int64_t{9}});
        (void)session.drain_commands();
        if (!expect(sink->last_value == "9", "caching-off fallback did not propagate the new value")) {
            return 1;
        }
        if (!expect(session.generation() == 2, "caching-off dataflow change did not bump the epoch")) {
            return 1;
        }
    }

    // run_once: headless start -> sweep -> stop.
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/true));
        const auto output = session.run_once();
        if (!expect(!output.has_value(), "sink-terminated run_once should return nullopt")) {
            return 1;
        }
        if (!expect(session.state() == leakflow::PipelineSessionState::Stopped, "run_once did not end stopped")) {
            return 1;
        }
        if (!expect(sink->process_count == 1, "run_once did not process once")) {
            return 1;
        }
    }

    // link_caps_error returns nullopt for a valid graph (D4 validation hook).
    {
        std::shared_ptr<NumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        leakflow::PipelineSession session(make_chain(src, mid, sink, /*sink_replay=*/true));
        if (!expect(!session.pipeline().link_caps_error().has_value(), "valid graph reported a caps error")) {
            return 1;
        }
    }

    // Liveness-aware property change (S11.5): a dataflow change on a live-driven
    // element applies FORWARD -- it must NOT re-emit from cache (no rerun), unlike
    // the one-run case which does.
    {
        std::shared_ptr<LiveNumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        leakflow::PipelineSession session(make_live_chain(src, mid, sink, /*sink_replay=*/false));

        (void)session.run_sweep();
        if (!expect(sink->process_count == 1, "initial live sweep did not process the sink once")) {
            return 1;
        }

        session.submit({"livesrc", "value", std::int64_t{7}});
        const auto applied = session.drain_commands();
        if (!expect(applied == 1, "live drain did not apply one command")) {
            return 1;
        }
        // Forward: NO rerun -> the sink was not reprocessed from cache.
        if (!expect(sink->process_count == 1, "live-driven change wrongly re-emitted from cache")) {
            return 1;
        }
        // It is still a config change -> the generation advances.
        if (!expect(session.generation() == 2, "live-driven dataflow change did not bump the generation")) {
            return 1;
        }
    }

    // Idle after a live run is held data, not a stream with a future buffer. A
    // live-driven property edit must therefore replay cached data, and replaying
    // through a non-replayable downstream element is rejected transactionally.
    {
        std::shared_ptr<LiveNumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        auto observer = std::make_shared<RecordingObserver>();
        leakflow::PipelineSession session(make_live_chain(src, mid, sink, /*sink_replay=*/false));
        session.set_observer(observer);

        (void)session.run_sweep();
        session.set_state(leakflow::PipelineSessionState::Idle);
        session.submit({"livesrc", "value", std::int64_t{7}});
        (void)session.drain_commands();

        if (!expect(src->property_as<std::int64_t>("value") == std::optional<std::int64_t>{0},
                    "Idle replay rejection mutated the live source property")) {
            return 1;
        }
        if (!expect(sink->last_value == "0", "Idle replay rejection propagated a new value")) {
            return 1;
        }
        if (!expect(sink->process_count == 1, "Idle replay rejection reprocessed the sink")) {
            return 1;
        }
        if (!expect(session.generation() == 1, "Idle replay rejection bumped the generation")) {
            return 1;
        }
        if (!expect(observer->rejected == 1 && observer->accepted == 0 && observer->applied == 0,
                    "Idle non-replayable change did not emit only a rejection")) {
            return 1;
        }
        if (!expect(observer->last_rejection_detail.find("sink") != std::string::npos,
                    "Idle non-replayable rejection did not identify the blocking element")) {
            return 1;
        }
    }

    // The same Idle live-driven edit is safe when every element in the replay-set
    // can replay; it recomputes the held frame and bumps the configuration
    // generation.
    {
        std::shared_ptr<LiveNumberSource> src;
        std::shared_ptr<PassThrough> mid;
        std::shared_ptr<CapturingSink> sink;
        auto observer = std::make_shared<RecordingObserver>();
        leakflow::PipelineSession session(make_live_chain(src, mid, sink, /*sink_replay=*/true));
        session.set_observer(observer);

        (void)session.run_sweep();
        session.set_state(leakflow::PipelineSessionState::Idle);
        session.submit({"livesrc", "value", std::int64_t{7}});
        (void)session.drain_commands();

        if (!expect(src->property_as<std::int64_t>("value") == std::optional<std::int64_t>{7},
                    "Idle replayable change did not mutate the live source property")) {
            return 1;
        }
        if (!expect(sink->last_value == "7", "Idle replayable change did not recompute the held value")) {
            return 1;
        }
        if (!expect(sink->process_count == 2, "Idle replayable change did not reprocess the sink")) {
            return 1;
        }
        if (!expect(session.generation() == 2, "Idle replayable dataflow change did not bump the generation")) {
            return 1;
        }
        if (!expect(observer->accepted == 1 && observer->applied == 1 && observer->rejected == 0,
                    "Idle replayable change did not emit accepted+applied")) {
            return 1;
        }
    }

    return 0;
}
