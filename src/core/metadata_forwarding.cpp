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

// Apply the per-buffer forwarding rules of profile into output for one input pad,
// using capture_values to detect conflicting capture facts across inputs.
void forward_one(
    std::string_view pad_name,
    const Buffer& input,
    ForwardingProfile profile,
    Buffer& output,
    std::map<std::string, std::string>& capture_values)
{
    const bool copy_payload = profile == ForwardingProfile::PassThrough;
    const bool prefix_origin = profile == ForwardingProfile::Analyze;

    for (const auto& [key, value] : input.metadata()) {
        switch (metadata_group(key)) {
        case MetadataGroup::Capture: {
            const auto seen = capture_values.find(key);
            if (seen == capture_values.end()) {
                capture_values.emplace(key, value);
                output.set_metadata(key, value);
            } else if (seen->second != value) {
                throw std::invalid_argument(
                    "conflicting capture metadata '" + key + "': '" + seen->second
                    + "' vs '" + value + "'");
            }
            break;
        }
        case MetadataGroup::Origin:
            if (prefix_origin) {
                // Flatten already-forwarded origin (origin.<prev>.<...>) instead of
                // nesting a second "origin." segment, yielding a readable provenance
                // path such as origin.targets.keys.file.path.
                std::string_view rest = key;
                if (rest.starts_with("origin.")) {
                    rest.remove_prefix(std::string_view("origin.").size());
                }
                output.set_metadata("origin." + std::string(pad_name) + "." + std::string(rest), value);
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

void forward_metadata(const ElementInputs& inputs, ForwardingProfile profile, Buffer& output)
{
    if (profile == ForwardingProfile::Source || profile == ForwardingProfile::Sink) {
        return;
    }

    std::map<std::string, std::string> capture_values;
    for (const auto& [pad_name, maybe_buffer] : inputs) {
        if (maybe_buffer) {
            forward_one(pad_name, *maybe_buffer, profile, output, capture_values);
        }
    }
}

void forward_metadata(
    const Buffer& input,
    ForwardingProfile profile,
    Buffer& output,
    std::string_view pad_name)
{
    if (profile == ForwardingProfile::Source || profile == ForwardingProfile::Sink) {
        return;
    }

    std::map<std::string, std::string> capture_values;
    forward_one(pad_name, input, profile, output, capture_values);
}

} // namespace leakflow
