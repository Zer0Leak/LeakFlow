#include "leakflow/core/element.hpp"
#include "leakflow/core/progress_sink.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <utility>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

class ProbeElement final : public leakflow::Element {
public:
    explicit ProbeElement(std::string name)
        : Element(std::move(name))
    {
    }

    void start() override
    {
        started = true;
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        processed = true;
        saw_input = input.has_value();

        if (input) {
            input->set_metadata("processed", "yes");
        }

        return input;
    }

    void stop() override
    {
        stopped = true;
    }

    // report_progress is protected; expose it so the progress-sink test can drive it.
    void emit_progress(double fraction, std::string message,
        leakflow::ProgressStatus status = leakflow::ProgressStatus::Active)
    {
        report_progress(fraction, std::move(message), 0, 0, status);
    }

    bool started = false;
    bool processed = false;
    bool saw_input = false;
    bool stopped = false;
};

class CapturingProgressSink final : public leakflow::ProgressSink {
public:
    void report(leakflow::Element& element, const leakflow::ElementProgress& progress) override
    {
        ++count;
        last_element = element.name();
        last = progress;
    }

    int count = 0;
    std::string last_element;
    leakflow::ElementProgress last;
};

} // namespace

int main()
{
    ProbeElement element("probe");

    if (!expect(element.name() == "probe", "name storage failed")) {
        return 1;
    }

    element.start();
    if (!expect(element.started, "start override was not called")) {
        return 1;
    }

    auto output = element.process(leakflow::Buffer(leakflow::Caps("sca/traceset")));
    if (!expect(element.processed, "process override was not called")) {
        return 1;
    }
    if (!expect(element.saw_input, "process did not receive buffer")) {
        return 1;
    }
    if (!expect(output.has_value(), "process did not return buffer")) {
        return 1;
    }
    if (!expect(output->metadata("processed") == "yes", "process did not update buffer")) {
        return 1;
    }

    element.stop();
    if (!expect(element.stopped, "stop override was not called")) {
        return 1;
    }

    // Progress: no sink -> report_progress is a silent no-op (must not crash).
    element.emit_progress(0.5, "ignored");

    // With a sink, the first report forwards; an immediate second is coalesced (< ~30 Hz); a
    // completion (fraction 1.0) always flushes regardless of the throttle window.
    CapturingProgressSink sink;
    element.set_progress_sink(&sink);
    element.emit_progress(0.5, "half");
    if (!expect(sink.count == 1 && sink.last_element == "probe", "first progress report did not forward")) {
        return 1;
    }
    if (!expect(sink.last.fraction == 0.5 && sink.last.message == "half"
            && sink.last.status == leakflow::ProgressStatus::Active,
            "progress payload wrong")) {
        return 1;
    }
    element.emit_progress(0.6, "coalesced");
    if (!expect(sink.count == 1, "immediate second report should be coalesced")) {
        return 1;
    }
    element.emit_progress(1.0, "done");
    if (!expect(sink.count == 2 && sink.last.fraction == 1.0
            && sink.last.status == leakflow::ProgressStatus::Completed,
            "completion should always flush and infer Completed status")) {
        return 1;
    }
    element.emit_progress(0.4, "cancelled", leakflow::ProgressStatus::Cancelled);
    if (!expect(sink.count == 3 && sink.last.fraction == 1.0
            && sink.last.status == leakflow::ProgressStatus::Cancelled,
            "cancellation should always flush as terminal 100%")) {
        return 1;
    }

    return 0;
}
