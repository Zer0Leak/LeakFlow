#pragma once

#include "leakflow/core/pad.hpp"
#include "leakflow/core/property.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

enum class ElementRank : int {
    None = 0,
    Marginal = 64,
    Secondary = 128,
    Primary = 256,
};

enum class MetadataPadTargetKind {
    AllPads,
    PadName,
    PadTemplate,
};

struct MetadataPadTarget {
    MetadataPadTargetKind kind = MetadataPadTargetKind::AllPads;
    std::optional<PadDirection> direction;
    std::string name;
};

struct ElementMetadataDescriptor {
    std::string key;
    PropertyValue value_type = std::string();
    std::string description;
    std::vector<std::string> example_values;
    std::string details;
    std::string value_hint;
    std::vector<MetadataPadTarget> pad_targets;
};

struct ElementDescriptor {
    std::string type_name;
    std::string long_name;
    ElementRank rank = ElementRank::None;
    std::string klass;
    std::string purpose;
    std::vector<Pad> pad_templates;
    std::vector<Pad> input_pads;
    std::vector<Pad> output_pads;
    std::vector<PropertySpec> property_specs;
    std::vector<std::string> keywords;
    std::vector<ElementMetadataDescriptor> metadata_set_by_element;
    std::vector<ElementMetadataDescriptor> metadata_suggestions;
    // Replay-safety for inspection (Phase 25). Mirrors Element::can_replay() so
    // leakflow-ls can show it. Stateful elements set this false.
    bool can_replay = true;
    // Vector-clock provenance slots claimed at add() (Phase 27). 1 for most
    // producers; 0 for pure fan-out (Tee) and sinks that need no own generation
    // (a 0-slot element forwards provenance verbatim, so fan-out branches stay
    // identical and match at a rejoin). See docs/design/dataflow_sync_model.md.
    int provenance_slots = 1;
    // Live-source flag (live phase). A live source streams many buffers over time
    // (capture/streaming); default false = one-run (file/static). Used to detect
    // live mode (run() pumps) and to drive liveness-aware property changes. See
    // docs/design/dataflow_sync_model.md S11.5.
    bool live_source = false;
    // Thread-boundary flag (live phase). A Queue cuts the pipeline into segments:
    // the producing segment pushes into it, the consuming segment pulls from it, on
    // separate threads. Segment decomposition removes thread-boundary elements and
    // splits the graph there. Default false (normal in-segment element). See
    // docs/design/dataflow_sync_model.md S10.
    bool thread_boundary = false;
};

[[nodiscard]] ElementMetadataDescriptor make_element_metadata_descriptor(std::string key,
    PropertyValue value_type,
    std::string description,
    std::vector<std::string> example_values = {},
    std::string details = {},
    std::string value_hint = {},
    std::vector<MetadataPadTarget> pad_targets = {});

[[nodiscard]] MetadataPadTarget metadata_all_pads();
[[nodiscard]] MetadataPadTarget metadata_all_pads(PadDirection direction);
[[nodiscard]] MetadataPadTarget metadata_pad(std::string name);
[[nodiscard]] MetadataPadTarget metadata_pad(PadDirection direction, std::string name);
[[nodiscard]] MetadataPadTarget metadata_pad_template(std::string name);
[[nodiscard]] MetadataPadTarget metadata_pad_template(PadDirection direction, std::string name);
[[nodiscard]] bool metadata_pad_template_matches(std::string_view pad_template, std::string_view pad_name);

ElementDescriptor& register_metadata_set_by_element(
    ElementDescriptor& descriptor,
    ElementMetadataDescriptor metadata);

ElementDescriptor& register_metadata_suggestion(
    ElementDescriptor& descriptor,
    ElementMetadataDescriptor metadata);

struct PluginDescriptor {
    std::string name;
    std::string owner;
    std::string author;
    std::string license;
    std::string version;
    std::string purpose;
    std::vector<std::string> keywords;
    std::vector<ElementDescriptor> elements;
};

[[nodiscard]] std::string element_default_name_prefix(std::string_view type_name);
[[nodiscard]] PropertySpec element_name_property_spec(std::string default_name);
[[nodiscard]] ElementDescriptor with_common_element_properties(ElementDescriptor descriptor);
[[nodiscard]] PluginDescriptor with_common_element_properties(PluginDescriptor descriptor);

} // namespace leakflow
