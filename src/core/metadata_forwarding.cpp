#include "leakflow/core/metadata_forwarding.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace leakflow {
namespace {

[[nodiscard]] std::string_view leading_segment(std::string_view text, char separator)
{
    const auto position = text.find(separator);
    if (position == std::string_view::npos) {
        return text;
    }

    return text.substr(0, position);
}

} // namespace

MetadataGroup metadata_group(std::string_view key)
{
    // Every stamped metadata key carries its group as the leading segment.
    const auto prefix = leading_segment(key, '.');
    if (prefix == "capture") {
        return MetadataGroup::Capture;
    }
    if (prefix == "origin") {
        return MetadataGroup::Origin;
    }
    if (prefix == "routing") {
        return MetadataGroup::Routing;
    }

    // "payload.*" and any unprefixed/unknown key resolve to payload, the safe
    // default for facts that should not be forwarded onto derived buffers.
    return MetadataGroup::Payload;
}

ForwardingProfile profile_for_klass(std::string_view klass)
{
    const auto head = leading_segment(klass, '/');
    if (head == "Source") {
        return ForwardingProfile::Source;
    }
    if (head == "Sink") {
        return ForwardingProfile::Sink;
    }
    if (head == "PassThrough") {
        return ForwardingProfile::PassThrough;
    }
    if (head == "Convert") {
        return ForwardingProfile::Reframe;
    }
    if (head == "Analyze") {
        return ForwardingProfile::Analyze;
    }

    // Unknown / future klasses (e.g. Control): be conservative and copy through.
    return ForwardingProfile::PassThrough;
}

namespace {

struct CaptureValue {
    std::string pad_name;
    std::string value;
};

// Relabel a forwarded fact as provenance of one input pad: origin.<pad>.<key>,
// stripping a redundant leading "origin." so an already-forwarded origin key stays
// flat (origin.<pad>.keys.file.path, not origin.<pad>.origin.keys.file.path).
[[nodiscard]] std::string prefixed_origin_key(std::string_view pad_name, std::string_view key)
{
    std::string_view rest = key;
    if (rest.starts_with("origin.")) {
        rest.remove_prefix(std::string_view("origin.").size());
    }
    return "origin." + std::string(pad_name) + "." + std::string(rest);
}

// Apply the per-buffer forwarding rules of profile into output for one input pad,
// using capture_values to detect conflicting capture facts across inputs.
void forward_one(
    std::string_view pad_name,
    const Buffer& input,
    ForwardingProfile profile,
    Buffer& output,
    std::map<std::string, CaptureValue>& capture_values,
    std::string_view element_name,
    bool is_reference)
{
    const bool copy_payload = profile == ForwardingProfile::PassThrough;
    const bool prefix_origin = profile == ForwardingProfile::Analyze;

    for (const auto& [key, value] : input.metadata()) {
        const auto group = metadata_group(key);

        // A Reference input carries a parameter from another experiment: the output
        // is guided by it but is not *about* it. Forward its durable facts (capture
        // and origin alike) as provenance under origin.<pad>.<key>, so they never
        // join -- or conflict with -- the output's own capture identity. Payload and
        // routing drop, as they would for any fused input.
        if (is_reference) {
            if (group == MetadataGroup::Capture || group == MetadataGroup::Origin) {
                output.set_metadata(prefixed_origin_key(pad_name, key), value);
            }
            continue;
        }

        switch (group) {
        case MetadataGroup::Capture: {
            const auto seen = capture_values.find(key);
            if (seen == capture_values.end()) {
                capture_values.emplace(key, CaptureValue{std::string(pad_name), value});
                output.set_metadata(key, value);
            } else if (seen->second.value != value) {
                auto message = std::string("metadata forwarding");
                if (!element_name.empty()) {
                    message += " in element '" + std::string(element_name) + "'";
                }
                message += ": conflicting capture metadata '" + key + "': input pad '"
                    + seen->second.pad_name + "' has '" + seen->second.value
                    + "', input pad '" + std::string(pad_name) + "' has '" + value + "'";
                throw std::invalid_argument(std::move(message));
            }
            break;
        }
        case MetadataGroup::Origin:
            if (prefix_origin) {
                output.set_metadata(prefixed_origin_key(pad_name, key), value);
            } else {
                output.set_metadata(key, value);
            }
            break;
        case MetadataGroup::Payload:
            if (copy_payload) {
                output.set_metadata(key, value);
            }
            break;
        case MetadataGroup::Routing:
            // Never forwarded; stamped link-locally by producers.
            break;
        }
    }
}

} // namespace

void forward_metadata(
    const ElementInputs& inputs,
    ForwardingProfile profile,
    Buffer& output,
    std::string_view element_name,
    const std::set<std::string_view>& reference_pads)
{
    if (profile == ForwardingProfile::Source || profile == ForwardingProfile::Sink) {
        return;
    }

    std::map<std::string, CaptureValue> capture_values;
    for (const auto& [pad_name, maybe_buffer] : inputs) {
        if (maybe_buffer) {
            const bool is_reference = reference_pads.contains(std::string_view(pad_name));
            forward_one(pad_name, *maybe_buffer, profile, output, capture_values, element_name, is_reference);
        }
    }
}

void forward_metadata(const Element& element, const ElementInputs& inputs, Buffer& output)
{
    std::set<std::string_view> reference_pads;
    for (const auto& pad : element.input_pads()) {
        if (pad.metadata_role() == PadMetadataRole::Reference) {
            reference_pads.emplace(pad.name());
        }
    }
    forward_metadata(inputs, profile_for_klass(element.element_kclass()), output, element.name(), reference_pads);
}

void forward_metadata(
    const Buffer& input,
    ForwardingProfile profile,
    Buffer& output,
    std::string_view pad_name,
    std::string_view element_name)
{
    if (profile == ForwardingProfile::Source || profile == ForwardingProfile::Sink) {
        return;
    }

    std::map<std::string, CaptureValue> capture_values;
    forward_one(pad_name, input, profile, output, capture_values, element_name, /*is_reference=*/false);
}

} // namespace leakflow
