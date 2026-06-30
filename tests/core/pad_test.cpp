#include "leakflow/core/pad.hpp"

#include <iostream>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

} // namespace

int main()
{
    leakflow::Pad input_pad("traces", leakflow::PadDirection::Input, leakflow::Caps("sca/traceset", {
        {"dtype", "float32"},
        {"layout", "trace,sample"},
    }));

    if (!expect(input_pad.name() == "traces", "input pad name was not preserved")) {
        return 1;
    }
    if (!expect(input_pad.direction() == leakflow::PadDirection::Input, "input pad direction was not preserved")) {
        return 1;
    }
    if (!expect(input_pad.caps().type() == "sca/traceset", "input pad caps type was not preserved")) {
        return 1;
    }
    if (!expect(input_pad.caps().param("dtype") == "float32", "input pad caps dtype was not preserved")) {
        return 1;
    }
    if (!expect(input_pad.caps().param("layout") == "trace,sample", "input pad caps layout was not preserved")) {
        return 1;
    }
    if (!expect(input_pad.presence() == leakflow::PadPresence::Required, "input pad default presence was not required")) {
        return 1;
    }
    if (!expect(input_pad.is_required(), "input pad did not report required")) {
        return 1;
    }

    leakflow::Pad output_pad("labels", leakflow::PadDirection::Output, leakflow::Caps("sca/labels", {
        {"dtype", "uint8"},
    }));

    if (!expect(output_pad.name() == "labels", "output pad name was not preserved")) {
        return 1;
    }
    if (!expect(output_pad.direction() == leakflow::PadDirection::Output, "output pad direction was not preserved")) {
        return 1;
    }
    if (!expect(output_pad.caps().type() == "sca/labels", "output pad caps type was not preserved")) {
        return 1;
    }
    if (!expect(output_pad.caps().param("dtype") == "uint8", "output pad caps params were not preserved")) {
        return 1;
    }

    leakflow::Pad optional_pad(
        "traces",
        leakflow::PadDirection::Input,
        leakflow::Caps("leakflow/torch-tensor"),
        leakflow::PadPresence::Optional);
    if (!expect(optional_pad.presence() == leakflow::PadPresence::Optional,
            "optional pad presence was not preserved")) {
        return 1;
    }
    if (!expect(!optional_pad.is_required(), "optional pad reported required")) {
        return 1;
    }

    leakflow::Pad request_pad(
        "src_%u",
        leakflow::PadDirection::Output,
        leakflow::Caps(leakflow::any_caps_type),
        leakflow::PadPresence::OnRequest);
    if (!expect(request_pad.presence() == leakflow::PadPresence::OnRequest,
            "request pad presence was not preserved")) {
        return 1;
    }
    if (!expect(!request_pad.is_required(), "request pad reported required")) {
        return 1;
    }

    return 0;
}
