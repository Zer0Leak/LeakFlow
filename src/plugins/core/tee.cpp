#include "leakflow/plugins/core/tee.hpp"

#include <string>
#include <utility>

namespace leakflow::plugins::core {

ElementDescriptor Tee::descriptor()
{
    return {
        .type_name = "Tee",
        .klass = "PassThrough/Flow/Tee",
        .purpose = "fork one input buffer to multiple output branches",
        .pad_templates = {
            Pad("sink", PadDirection::Input, Caps(any_caps_type)),
            Pad("src_%u", PadDirection::Output, Caps(any_caps_type), PadPresence::OnRequest),
        },
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(any_caps_type)),
        },
        .property_specs = {},
        .keywords = {"tee", "branch", "fork"},
        .metadata_suggestions = {
            make_element_metadata_descriptor(
                "routing.branch",
                std::string(),
                "branch label stamped by user metadata on Tee outputs",
                {"plot", "sink", "poi"},
                "Applies to existing Tee source pads matching the src_%u template.",
                {},
                {metadata_pad_template(PadDirection::Output, "src_%u")}),
        },
        // Pure fan-out: claims no provenance slot, so every branch carries the
        // input buffer's provenance verbatim and matches at a downstream rejoin.
        .provenance_slots = 0,
    };
}

Tee::Tee(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> Tee::process(std::optional<Buffer> input)
{
    auto record = make_log_record(log::LogLevel::Debug, "element", "passed buffer to branches");
    record.fields.emplace("has_input", input ? "true" : "false");
    record.fields.emplace("branches", std::to_string(output_pads().size()));
    leakflow::log::write(std::move(record));
    return input;
}

ElementOutputs Tee::process_pads(ElementInputs inputs)
{
    std::optional<Buffer> input;
    if (!inputs.empty()) {
        input = std::move(inputs.begin()->second);
    }

    auto record = make_log_record(log::LogLevel::Debug, "element", "passed buffer to branches");
    record.fields.emplace("has_input", input ? "true" : "false");
    record.fields.emplace("branches", std::to_string(output_pads().size()));
    leakflow::log::write(std::move(record));

    // Fan-out is a Tee behavior, not an engine rule: copy the envelope to each
    // output pad (the payload shared_ptr is shared across branches).
    ElementOutputs outputs;
    if (input) {
        for (const auto& pad : output_pads()) {
            outputs.emplace(pad.name(), *input);
        }
    }
    return outputs;
}

std::vector<Buffer> Tee::fork(const Buffer& input) const
{
    std::vector<Buffer> branches;
    branches.reserve(output_pads().size());
    for (const auto& pad : output_pads()) {
        (void)pad;
        branches.push_back(input);
    }
    return branches;
}

} // namespace leakflow::plugins::core
