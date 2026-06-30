#include "leakflow/core/element.hpp"

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

    bool started = false;
    bool processed = false;
    bool saw_input = false;
    bool stopped = false;
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

    return 0;
}
