#include "leakflow/core/pipeline.hpp"
#include "leakflow/plugins/core/sync.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
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

std::uint32_t slot(const std::vector<std::uint32_t>& clock, std::size_t index)
{
    return index < clock.size() ? clock[index] : 0u;
}

class GenSource final : public leakflow::Element {
public:
    explicit GenSource(std::string name)
        : Element(std::move(name))
    {
        add_output_pad(leakflow::Pad("src", leakflow::PadDirection::Output, leakflow::Caps(leakflow::any_caps_type)));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        return leakflow::Buffer(leakflow::Caps(leakflow::any_caps_type));
    }
};

class GenJoin final : public leakflow::Element {
public:
    GenJoin()
        : Element("join")
    {
        add_input_pad(leakflow::Pad("left", leakflow::PadDirection::Input, leakflow::Caps(leakflow::any_caps_type)));
        add_input_pad(leakflow::Pad("right", leakflow::PadDirection::Input, leakflow::Caps(leakflow::any_caps_type)));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps(leakflow::any_caps_type)));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        throw std::invalid_argument("GenJoin requires named inputs");
    }

    std::optional<leakflow::Buffer> process_inputs(leakflow::ElementInputs inputs) override
    {
        if (!inputs.contains("left") || !inputs.contains("right")) {
            throw std::invalid_argument("GenJoin missing inputs");
        }
        return leakflow::Buffer(leakflow::Caps(leakflow::any_caps_type));
    }
};

class CaptureSink final : public leakflow::Element {
public:
    CaptureSink()
        : Element("capture")
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps(leakflow::any_caps_type)));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            ++count;
            last_provenance = input->provenance();
        }
        return std::nullopt;
    }

    int count = 0;
    std::vector<std::uint32_t> last_provenance;
};

} // namespace

int main()
{
    // Walkthrough Example 3b: two independent sources paired by a Sync element.
    // Add order fixes slots: a=1, b=2, sync=3, join=4, capture=5.
    leakflow::Pipeline pipeline;
    auto a = pipeline.add(std::make_shared<GenSource>("a"));
    auto b = pipeline.add(std::make_shared<GenSource>("b"));
    auto sync = pipeline.add(std::make_shared<leakflow::plugins::core::Sync>("sync"));
    auto join = pipeline.add(std::make_shared<GenJoin>());
    auto capture_element = std::make_shared<CaptureSink>();
    auto capture = pipeline.add(capture_element);

    pipeline.link(a, "src", sync, "in_0");
    pipeline.link(b, "src", sync, "in_1");
    pipeline.link(sync, "out_0", join, "left");
    pipeline.link(sync, "out_1", join, "right");
    pipeline.link(join, "out", capture, "in");

    (void)pipeline.run();

    if (!expect(capture_element->count == 1, "Sync pipeline did not deliver a buffer to the sink")) {
        return 1;
    }

    const auto& clock = capture_element->last_provenance;
    // The joined buffer carries Sync's slot (3): the Sync element injected a common
    // ancestor, so the downstream join matched on it (otherwise the two branches
    // would be independent). This is the proof of S11.4.
    if (!expect(slot(clock, 3) == 1,
            "Sync slot not stamped on the aligned outputs (common ancestor not injected)")) {
        return 1;
    }
    if (!expect(slot(clock, 1) == 1 && slot(clock, 2) == 1, "source slots were not preserved through Sync")) {
        return 1;
    }
    if (!expect(slot(clock, 4) == 1, "join slot was wrong")) {
        return 1;
    }

    return 0;
}
