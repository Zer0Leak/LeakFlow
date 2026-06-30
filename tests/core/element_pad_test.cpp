#include "leakflow/core/element.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
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

class DeclaringElement final : public leakflow::Element {
public:
    explicit DeclaringElement(std::string name)
        : Element(std::move(name))
    {
    }

    std::optional<leakflow::Buffer> process(std::optional<leakflow::Buffer> input) override
    {
        return input;
    }
};

bool throws_invalid_argument_for_input_pad(DeclaringElement& element, leakflow::Pad pad)
{
    try {
        element.add_input_pad(std::move(pad));
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

bool throws_invalid_argument_for_output_pad(DeclaringElement& element, leakflow::Pad pad)
{
    try {
        element.add_output_pad(std::move(pad));
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

} // namespace

int main()
{
    DeclaringElement element("declaring");

    if (!expect(element.input_pads().empty(), "new element should have zero input pads")) {
        return 1;
    }
    if (!expect(element.output_pads().empty(), "new element should have zero output pads")) {
        return 1;
    }

    element.add_input_pad(leakflow::Pad("traces", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset", {
        {"dtype", "float32"},
        {"layout", "trace,sample"},
    })));

    if (!expect(element.input_pads().size() == 1, "input pad was not stored")) {
        return 1;
    }
    if (!expect(element.input_pads()[0].name() == "traces", "input pad name was not preserved")) {
        return 1;
    }
    if (!expect(element.input_pads()[0].direction() == leakflow::PadDirection::Input,
            "input pad direction was not preserved")) {
        return 1;
    }
    if (!expect(element.input_pads()[0].caps().type() == "sca/traceset", "input pad caps type was not preserved")) {
        return 1;
    }
    if (!expect(element.input_pads()[0].caps().param("dtype") == "float32",
            "input pad caps dtype was not preserved")) {
        return 1;
    }
    if (!expect(element.input_pads()[0].caps().param("layout") == "trace,sample",
            "input pad caps layout was not preserved")) {
        return 1;
    }

    element.add_output_pad(leakflow::Pad("labels", leakflow::PadDirection::Output, leakflow::Caps("sca/labels", {
        {"dtype", "uint8"},
    })));

    if (!expect(element.output_pads().size() == 1, "output pad was not stored")) {
        return 1;
    }
    if (!expect(element.output_pads()[0].name() == "labels", "output pad name was not preserved")) {
        return 1;
    }
    if (!expect(element.output_pads()[0].direction() == leakflow::PadDirection::Output,
            "output pad direction was not preserved")) {
        return 1;
    }
    if (!expect(element.output_pads()[0].caps().type() == "sca/labels", "output pad caps type was not preserved")) {
        return 1;
    }
    if (!expect(element.output_pads()[0].caps().param("dtype") == "uint8",
            "output pad caps dtype was not preserved")) {
        return 1;
    }

    if (!expect(throws_invalid_argument_for_input_pad(
            element, leakflow::Pad("bad-input", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset"))),
            "add_input_pad did not reject output pad")) {
        return 1;
    }
    if (!expect(throws_invalid_argument_for_output_pad(
            element, leakflow::Pad("bad-output", leakflow::PadDirection::Input, leakflow::Caps("sca/labels"))),
            "add_output_pad did not reject input pad")) {
        return 1;
    }

    if (!expect(throws_invalid_argument_for_input_pad(
            element, leakflow::Pad("traces", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset"))),
            "duplicate input pad name was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument_for_output_pad(
            element, leakflow::Pad("labels", leakflow::PadDirection::Output, leakflow::Caps("sca/labels"))),
            "duplicate output pad name was not rejected")) {
        return 1;
    }

    element.add_output_pad(leakflow::Pad("traces", leakflow::PadDirection::Output, leakflow::Caps("sca/traceset")));
    element.add_input_pad(leakflow::Pad("labels", leakflow::PadDirection::Input, leakflow::Caps("sca/labels")));

    if (!expect(element.output_pads().size() == 2, "same name as input was not allowed for output pad")) {
        return 1;
    }
    if (!expect(element.input_pads().size() == 2, "same name as output was not allowed for input pad")) {
        return 1;
    }

    element.add_input_pad(leakflow::Pad(
        "optional_traces",
        leakflow::PadDirection::Input,
        leakflow::Caps("leakflow/torch-tensor"),
        leakflow::PadPresence::Optional));
    if (!expect(element.input_pads().size() == 3, "optional input pad was not stored")) {
        return 1;
    }
    if (!expect(element.input_pads()[2].presence() == leakflow::PadPresence::Optional,
            "optional input pad presence was not preserved")) {
        return 1;
    }

    element.add_pad_template(leakflow::Pad(
        "src_%u",
        leakflow::PadDirection::Output,
        leakflow::Caps(leakflow::any_caps_type),
        leakflow::PadPresence::OnRequest));
    if (!expect(element.pad_templates().size() == 1, "pad template was not stored")) {
        return 1;
    }
    if (!expect(element.can_request_output_pad("src_0"), "requestable output pad was not recognized")) {
        return 1;
    }
    if (!expect(!element.can_request_output_pad("src_last"),
            "requestable output pad matched a non-numeric suffix")) {
        return 1;
    }
    if (!expect(element.request_output_pad("src_0"), "requestable output pad was not created")) {
        return 1;
    }
    if (!expect(element.output_pads().size() == 3, "requested output pad was not stored")) {
        return 1;
    }
    if (!expect(element.output_pads()[2].name() == "src_0", "requested output pad name was wrong")) {
        return 1;
    }
    if (!expect(element.output_pads()[2].caps().type() == leakflow::any_caps_type,
            "requested output pad caps did not come from the template")) {
        return 1;
    }
    if (!expect(element.request_output_pad("src_0"), "requesting an existing output pad should succeed")) {
        return 1;
    }
    if (!expect(element.output_pads().size() == 3, "requesting an existing output pad created a duplicate")) {
        return 1;
    }
    if (!expect(!element.request_output_pad("missing_0"), "non-template output pad was unexpectedly created")) {
        return 1;
    }

    return 0;
}
