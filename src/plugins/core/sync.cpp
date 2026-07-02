#include "leakflow/plugins/core/sync.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow::plugins::core {

ElementDescriptor Sync::descriptor()
{
    return {
        .type_name = "Sync",
        .klass = "Sync/Branch",
        .purpose = "synchronize N independent input streams into N aligned outputs by injecting a common ancestor",
        .pad_templates =
            {
                Pad("in_%u", PadDirection::Input, Caps(any_caps_type), PadPresence::OnRequest),
                Pad("out_%u", PadDirection::Output, Caps(any_caps_type), PadPresence::OnRequest),
            },
        .property_specs =
            {
                PropertySpec("policy", std::string("all-required-once"),
                             "stream pairing policy; only all-required-once is wired offline", "",
                             StringEnumConstraint{{"all-required-once", "zip", "held", "latest", "barrier"}}, ""),
            },
        .keywords = {"sync", "synchronize", "zip", "barrier", "pair", "join"},
        // provenance_slots defaults to 1: Sync claims one slot and stamps every
        // aligned output of a firing with the same value (common-ancestor
        // injection, S11.4).
    };
}

Sync::Sync(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> Sync::process(std::optional<Buffer>)
{
    throw std::invalid_argument("Sync requires named input pads (in_0, in_1, ...)");
}

ElementOutputs Sync::process_pads(ElementInputs inputs)
{
    // AllRequiredOnce: the executor only calls us once all connected inputs are
    // present. Map in_K -> out_K. The executor then stamps every output with this
    // element's own slot (same value per firing), so out_0 and out_1 share the
    // Sync slot and become common-origin downstream.
    ElementOutputs outputs;
    for (auto& [pad_name, buffer] : inputs) {
        if (!buffer) {
            continue;
        }
        if (pad_name.rfind("in_", 0) == 0) {
            outputs.emplace("out_" + pad_name.substr(3), *buffer);
        }
    }

    auto record = make_log_record(log::LogLevel::Debug, "element", "synchronized inputs");
    record.fields.emplace("aligned_outputs", std::to_string(outputs.size()));
    leakflow::log::write(std::move(record));

    return outputs;
}

} // namespace leakflow::plugins::core
