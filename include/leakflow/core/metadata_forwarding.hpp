#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <string_view>

namespace leakflow {

// Taxonomy group a metadata key belongs to. The group determines how the key is
// forwarded across elements. See docs/design/metadata_klass_taxonomy.md.
enum class MetadataGroup {
    Capture, // durable acquisition/dataset/countermeasure facts
    Origin,  // provenance of one specific input (file + role)
    Payload, // producer's assertion about the current payload bytes
    Routing, // transient pipeline-topology / identity scratch
};

// Forwarding behaviour of an element, derived from the leading token of its klass.
enum class ForwardingProfile {
    Source,      // produces buffers; owns its metadata (no forwarding)
    PassThrough, // copies capture + origin + payload; drops routing
    Reframe,     // copies capture + origin; drops payload + routing (klass "Convert")
    Analyze,     // unions capture (conflict = error); origin -> origin.<pad>.<key>; drops payload + routing
    Sink,        // terminal; forwards nothing
};

// Classify a metadata key into its taxonomy group.
//
// Resolution is by the first dotted segment plus a small dotless table; every key
// the framework does not recognise (including all domain keys such as leakage.*,
// crypto.*, poi.*) falls through to Payload. This keeps leakflow_core free of any
// domain vocabulary.
[[nodiscard]] MetadataGroup metadata_group(std::string_view key);

// Map an element klass string to its forwarding profile via the leading klass token
// (Source / Sink / PassThrough / Convert / Analyze). Unknown klasses default to
// PassThrough, matching the conservative "copy everything" behaviour of generic
// forwarding elements.
[[nodiscard]] ForwardingProfile profile_for_klass(std::string_view klass);

// Forward metadata from the connected inputs into output according to the profile,
// before the element stamps its own keys:
//   - Source / Sink: no-op.
//   - PassThrough:    copy capture, origin, payload; drop routing.
//   - Reframe:        copy capture, origin; drop payload, routing.
//   - Analyze:        union capture (throws std::invalid_argument on conflicting
//                     values across inputs); relabel origin as origin.<pad>.<key>;
//                     drop payload, routing.
//
// Routing is never forwarded by any profile; producers stamp element/branch
// separately. Capture conflicts are reported with their (non-secret) values.
void forward_metadata(const ElementInputs& inputs, ForwardingProfile profile, Buffer& output);

// Convenience overload for single-input elements that hold one buffer (the common
// Reframe case). pad_name is only used by the Analyze profile for origin relabeling.
void forward_metadata(
    const Buffer& input,
    ForwardingProfile profile,
    Buffer& output,
    std::string_view pad_name = "in");

} // namespace leakflow
