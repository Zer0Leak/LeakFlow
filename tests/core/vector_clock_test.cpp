#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/provenance.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
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

template <typename Function>
bool throws_invalid_argument(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

std::uint32_t slot(const std::vector<std::uint32_t>& clock, std::size_t index)
{
    return index < clock.size() ? clock[index] : 0u;
}

// A source element: one output pad, claims one provenance slot (the default).
class ProvSource final : public leakflow::Element {
public:
    explicit ProvSource(std::string name)
        : Element(std::move(name))
    {
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            return input;
        }
        return leakflow::Buffer(leakflow::Caps("sca/traceset"));
    }
};

// A two-input join: merges left+right into one output (default process_pads
// routes it to the sole output pad; the executor stamps the merged provenance).
class ProvJoin final : public leakflow::Element {
public:
    ProvJoin()
        : Element("join")
    {
        add_input_pad(leakflow::Pad("left", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset")));
        add_input_pad(leakflow::Pad("right", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset")));
        add_output_pad(leakflow::Pad("out", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer>) override
    {
        throw std::invalid_argument("ProvJoin requires named inputs");
    }

    std::optional<leakflow::Buffer> process_inputs(leakflow::ElementInputs inputs) override
    {
        if (!inputs.contains("left") || !inputs.contains("right")) {
            throw std::invalid_argument("ProvJoin missing inputs");
        }
        return leakflow::Buffer(leakflow::Caps("sca/traceset"));
    }
};

// A sink that records the provenance of the last buffer it received.
class CaptureSink final : public leakflow::Element {
public:
    CaptureSink()
        : Element("capture")
    {
        add_input_pad(leakflow::Pad("in", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset")));
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        if (input) {
            last_provenance = input->provenance();
        }
        return std::nullopt;
    }

    std::vector<std::uint32_t> last_provenance;
};

// A 0-slot fan-out (mini-Tee): copies its input to both output pads, claiming no
// provenance slot, so both branches carry identical provenance.
class FanOut final : public leakflow::Element {
public:
    FanOut()
        : Element("fanout")
    {
        leakflow::ElementDescriptor descriptor;
        descriptor.type_name = "FanOut";
        descriptor.klass = "Test/Branch";
        descriptor.input_pads = {leakflow::Pad("sink", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset"))};
        descriptor.output_pads = {
            leakflow::Pad("a", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")),
            leakflow::Pad("b", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")),
        };
        descriptor.provenance_slots = 0;
        configure_from_descriptor(descriptor);
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        return input;
    }

    leakflow::ElementOutputs process_pads(leakflow::ElementInputs inputs) override
    {
        std::optional<leakflow::Buffer> input;
        if (!inputs.empty()) {
            input = std::move(inputs.begin()->second);
        }
        leakflow::ElementOutputs outputs;
        if (input) {
            outputs.emplace("a", *input);
            outputs.emplace("b", *input);
        }
        return outputs;
    }
};

bool test_provenance_functions()
{
    // Compatibility: zeros are wildcards.
    if (!expect(leakflow::provenance_compatible({0, 5, 0}, {0, 5, 7}), "equal-overlap clocks were incompatible")) {
        return false;
    }
    if (!expect(leakflow::provenance_compatible({0, 5, 0}, {0, 0, 7}), "wildcard clocks were incompatible")) {
        return false;
    }
    if (!expect(!leakflow::provenance_compatible({0, 5, 0}, {0, 6, 0}), "conflicting clocks were compatible")) {
        return false;
    }

    // Fold-and-detect: merge is component-wise max.
    const std::vector<std::uint32_t> a{0, 5, 0, 0};
    const std::vector<std::uint32_t> b{0, 5, 7, 0};
    const auto merged = leakflow::merge_provenance({&a, &b}, 4);
    if (!expect(merged == std::vector<std::uint32_t>{0, 5, 7, 0}, "merge did not produce the component-wise max")) {
        return false;
    }

    // N=3 conflict that pairwise-against-first would miss (index 2: 7 vs 8,
    // with the first clock zero there).
    const std::vector<std::uint32_t> c0{0, 5};
    const std::vector<std::uint32_t> c1{0, 5, 7};
    const std::vector<std::uint32_t> c2{0, 0, 8};
    if (!expect(throws_invalid_argument([&] { (void)leakflow::merge_provenance({&c0, &c1, &c2}, 3); }),
            "merge did not detect the N=3 cross-conflict")) {
        return false;
    }

    return true;
}

bool test_join_provenance()
{
    // Add order fixes the slots: src_a=1, src_b=2, join=3, capture=4.
    leakflow::Pipeline pipeline;
    auto src_a = pipeline.add(std::make_shared<ProvSource>("src_a"));
    auto src_b = pipeline.add(std::make_shared<ProvSource>("src_b"));
    auto join = pipeline.add(std::make_shared<ProvJoin>());
    auto capture_element = std::make_shared<CaptureSink>();
    auto capture = pipeline.add(capture_element);
    pipeline.link(src_a, "out", join, "left");
    pipeline.link(src_b, "out", join, "right");
    pipeline.link(join, "out", capture, "in");

    (void)pipeline.run();

    const auto& clock = capture_element->last_provenance;
    if (!expect(slot(clock, 1) == 1 && slot(clock, 2) == 1 && slot(clock, 3) == 1,
            "join output provenance did not carry both sources plus the join slot")) {
        return false;
    }
    if (!expect(slot(clock, 0) == 0 && slot(clock, 4) == 0,
            "join output provenance had unexpected non-zero slots")) {
        return false;
    }

    return true;
}

bool test_fanout_rejoin_matches()
{
    // src=1, fanout=0-slot, join=2, capture=3. Both branches carry src's clock
    // verbatim, so the rejoin folds without conflict.
    leakflow::Pipeline pipeline;
    auto src = pipeline.add(std::make_shared<ProvSource>("src"));
    auto fanout = pipeline.add(std::make_shared<FanOut>());
    auto join = pipeline.add(std::make_shared<ProvJoin>());
    auto capture_element = std::make_shared<CaptureSink>();
    auto capture = pipeline.add(capture_element);
    pipeline.link(src, "out", fanout, "sink");
    pipeline.link(fanout, "a", join, "left");
    pipeline.link(fanout, "b", join, "right");
    pipeline.link(join, "out", capture, "in");

    std::optional<leakflow::Buffer> result;
    if (!expect(!throws_invalid_argument([&] { result = pipeline.run(); }),
            "fan-out rejoin reported a spurious provenance conflict")) {
        return false;
    }

    const auto& clock = capture_element->last_provenance;
    // src=1 (carried through both branches), fanout has no slot, join=2.
    if (!expect(slot(clock, 1) == 1 && slot(clock, 2) == 1,
            "fan-out rejoin provenance was wrong")) {
        return false;
    }

    return true;
}

bool test_rerun_supersede()
{
    // src_a=1, src_b=2, join=3, capture=4. A partial rerun from the join must
    // bump the join's own slot (supersede) while sources stay from cache.
    leakflow::Pipeline pipeline;
    auto src_a = pipeline.add(std::make_shared<ProvSource>("src_a"));
    auto src_b = pipeline.add(std::make_shared<ProvSource>("src_b"));
    auto join = pipeline.add(std::make_shared<ProvJoin>());
    auto capture_element = std::make_shared<CaptureSink>();
    auto capture = pipeline.add(capture_element);
    pipeline.link(src_a, "out", join, "left");
    pipeline.link(src_b, "out", join, "right");
    pipeline.link(join, "out", capture, "in");

    pipeline.start_all();
    (void)pipeline.run_sweep();
    if (!expect(slot(capture_element->last_provenance, 3) == 1, "initial join slot was not 1")) {
        pipeline.stop_all();
        return false;
    }

    (void)pipeline.rerun_from(join);
    if (!expect(slot(capture_element->last_provenance, 3) == 2, "rerun did not supersede the join slot")) {
        pipeline.stop_all();
        return false;
    }
    // Sources stayed valid from cache (no new generation).
    if (!expect(slot(capture_element->last_provenance, 1) == 1 && slot(capture_element->last_provenance, 2) == 1,
            "rerun changed the cached source provenance")) {
        pipeline.stop_all();
        return false;
    }
    pipeline.stop_all();

    return true;
}

} // namespace

int main()
{
    if (!test_provenance_functions()) {
        return 1;
    }
    if (!test_join_provenance()) {
        return 1;
    }
    if (!test_fanout_rejoin_matches()) {
        return 1;
    }
    if (!test_rerun_supersede()) {
        return 1;
    }

    return 0;
}
