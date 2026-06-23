#pragma once

#include "leakflow/core/caps.hpp"
#include "leakflow/core/pad.hpp"
#include "leakflow/core/property.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

struct PipelinePadSnapshot {
    std::string name;
    PadDirection direction = PadDirection::Input;
    PadPresence presence = PadPresence::Required;
    Caps caps = Caps(any_caps_type);
};

struct PipelinePropertySnapshot {
    std::string name;
    std::string value_type;
    std::string value;
    PropertyEffect effect;
};

struct PipelineElementSnapshot {
    std::string type_name;
    std::string name;
    std::string klass;
    std::vector<PipelinePropertySnapshot> properties;
    std::vector<PipelinePadSnapshot> input_pads;
    std::vector<PipelinePadSnapshot> output_pads;
    std::vector<PipelinePadSnapshot> pad_templates;
    // Vector-clock provenance slots claimed by this element (Phase 27). base is
    // the first slot index; slots is how many (0 for Tee/sinks). UI labels slot
    // `base` with the element name (and `element.pad` once per-pad slots exist).
    std::uint32_t provenance_base = 0;
    int provenance_slots = 0;
};

struct PipelineEndpointSnapshot {
    std::string element_type;
    std::string element_name;
    std::string element_klass;
    std::string pad_name;
};

struct PipelineLinkSnapshot {
    std::string id;
    PipelineEndpointSnapshot source;
    PipelineEndpointSnapshot sink;
    Caps source_caps = Caps(any_caps_type);
    Caps sink_caps = Caps(any_caps_type);
};

struct PipelineTopologySnapshot {
    std::vector<PipelineElementSnapshot> elements;
    std::vector<PipelineLinkSnapshot> links;
};

struct PipelineBufferSnapshot {
    Caps caps = Caps(any_caps_type);
    std::map<std::string, std::string> metadata;
    bool has_payload = false;
    std::string payload_type;
    std::vector<std::string> payload_summary;
};

struct PipelineBufferObservation {
    std::string link_id;
    PipelineEndpointSnapshot source;
    PipelineEndpointSnapshot sink;
    PipelineBufferSnapshot buffer;
    std::uint64_t sequence = 0;
    // Scalar generation derived from the buffer's vector clock (Phase 27), for
    // UI change detection only; matching uses the full clock, not this.
    std::uint32_t generation = 0;
    // The buffer's full vector clock (copied), for the provenance/index panel.
    std::vector<std::uint32_t> provenance;
};

struct PipelinePropertyChangeObservation {
    PipelineEndpointSnapshot element;
    std::string property_name;
    std::string value_type;
    std::string previous_value;
    std::string new_value;
    PropertyEffect effect;
};

// Copied, SCA-safe result of a control-plane command (Phase 25). Carries no
// mutable handles and no payload data.
enum class PipelineCommandStatus {
    Accepted,
    Rejected,
    Applied,
    Failed,
};

struct PipelineCommandObservation {
    PipelineCommandStatus status = PipelineCommandStatus::Accepted;
    PipelineEndpointSnapshot element;
    std::string property_name;
    std::string value_type;
    std::string previous_value;
    std::string new_value;
    PropertyEffect effect;
    // Session control-plane generation counter (Phase 27, renamed from epoch).
    std::uint64_t generation = 0;
    std::string detail;
};

enum class PipelineEventKind {
    TopologySnapshot,
    Started,
    Stopped,
    Error,
    ElementStarted,
    ElementStopped,
    BufferObserved,
    PropertyChanged,
    CommandAccepted,
    CommandRejected,
    CommandApplied,
};

struct PipelineEvent {
    PipelineEventKind kind = PipelineEventKind::Started;
    std::uint64_t sequence = 0;
    std::optional<PipelineTopologySnapshot> topology;
    std::optional<PipelineBufferObservation> buffer;
    std::optional<PipelinePropertyChangeObservation> property_change;
    std::optional<PipelineCommandObservation> command;
    std::string element_name;
    std::string message;
};

class PipelineObserver {
public:
    virtual ~PipelineObserver() = default;
    virtual void observe(const PipelineEvent &event) = 0;
};

[[nodiscard]] std::string pipeline_link_id(std::string_view source_element_name, std::string_view source_pad_name,
                                           std::string_view sink_element_name, std::string_view sink_pad_name);

} // namespace leakflow
