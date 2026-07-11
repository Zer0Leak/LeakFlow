#include "common/terminal_pager.hpp"

#include "leakflow/core/descriptor_registry.hpp"
#include "leakflow/render/terminal_style.hpp"
#include "leakflow/plugins/base/descriptor_catalog.hpp"
#include "leakflow/plugins/core/descriptor_catalog.hpp"
#include "leakflow/plugins/crypto/descriptor_catalog.hpp"
#include "leakflow/plugins/crypto_plot/descriptor_catalog.hpp"
#include "leakflow/plugins/extras/descriptor_catalog.hpp"
#include "leakflow/plugins/ml/descriptor_catalog.hpp"
#include "leakflow/plugins/plot/descriptor_catalog.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <optional>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace core = leakflow::plugins::core;
namespace base = leakflow::plugins::base;
namespace ml_plugin = leakflow::plugins::ml;
namespace crypto_plugin = leakflow::plugins::crypto;
namespace crypto_plot_plugin = leakflow::plugins::crypto_plot;
namespace extras = leakflow::plugins::extras;
namespace plot_plugin = leakflow::plugins::plot;
namespace render = leakflow::render;

struct RenderContext {
    render::TerminalStyler styler;
    render::GlyphSet glyphs;
};

class CoutCapture {
public:
    CoutCapture()
        : previous_(std::cout.rdbuf(buffer_.rdbuf()))
    {
    }

    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;

    ~CoutCapture()
    {
        restore();
    }

    [[nodiscard]] std::string release()
    {
        restore();
        return buffer_.str();
    }

private:
    void restore()
    {
        if (previous_ != nullptr) {
            std::cout.rdbuf(previous_);
            previous_ = nullptr;
        }
    }

    std::ostringstream buffer_;
    std::streambuf* previous_ = nullptr;
};

[[nodiscard]] leakflow::DescriptorRegistry linked_descriptor_registry()
{
    leakflow::DescriptorRegistry registry;
    core::register_plugin_descriptors(registry);
    base::register_plugin_descriptors(registry);
    ml_plugin::register_plugin_descriptors(registry);
    extras::register_plugin_descriptors(registry);
    crypto_plugin::register_plugin_descriptors(registry);
    plot_plugin::register_plugin_descriptors(registry);
    crypto_plot_plugin::register_plugin_descriptors(registry);
    return registry;
}

[[nodiscard]] const leakflow::DescriptorRegistry& descriptor_registry()
{
    static const auto registry = linked_descriptor_registry();
    return registry;
}

[[nodiscard]] std::string styled(const RenderContext& context, render::StyleRole role, std::string_view text)
{
    return context.styler.apply(role, text);
}

[[nodiscard]] std::string pad_direction_name(leakflow::PadDirection direction)
{
    switch (direction) {
    case leakflow::PadDirection::Input:
        return "SINK";
    case leakflow::PadDirection::Output:
        return "SRC";
    }

    return "UNKNOWN";
}

[[nodiscard]] std::string pad_presence_name(leakflow::PadPresence presence)
{
    switch (presence) {
    case leakflow::PadPresence::Required:
        return "Always";
    case leakflow::PadPresence::Optional:
        return "Optional";
    case leakflow::PadPresence::OnRequest:
        return "On request";
    }

    return "Unknown";
}

[[nodiscard]] std::string element_klass(const leakflow::ElementDescriptor& element)
{
    if (!element.klass.empty()) {
        return element.klass;
    }

    if (element.input_pads.empty() && !element.output_pads.empty()) {
        return "Source";
    }
    if (!element.input_pads.empty() && element.output_pads.empty()) {
        return "Sink";
    }
    if (!element.input_pads.empty() && element.output_pads.size() > 1) {
        return "Generic/Branch";
    }
    if (!element.input_pads.empty() && !element.output_pads.empty()) {
        return "Generic";
    }

    return "Generic";
}

[[nodiscard]] std::string element_long_name(const leakflow::ElementDescriptor& element)
{
    if (!element.long_name.empty()) {
        return element.long_name;
    }

    return element.type_name;
}

[[nodiscard]] std::string element_rank_name(leakflow::ElementRank rank)
{
    switch (rank) {
    case leakflow::ElementRank::None:
        return "none";
    case leakflow::ElementRank::Marginal:
        return "marginal";
    case leakflow::ElementRank::Secondary:
        return "secondary";
    case leakflow::ElementRank::Primary:
        return "primary";
    }

    return "custom";
}

[[nodiscard]] std::string element_rank_text(leakflow::ElementRank rank)
{
    std::ostringstream output;
    output << element_rank_name(rank) << " (" << static_cast<int>(rank) << ')';
    return output.str();
}

[[nodiscard]] std::string element_flags(const leakflow::ElementDescriptor& element)
{
    const auto has_input = !element.input_pads.empty();
    auto has_output = !element.output_pads.empty();
    for (const auto& pad_template : element.pad_templates) {
        if (pad_template.direction() == leakflow::PadDirection::Output) {
            has_output = true;
            break;
        }
    }

    if (has_input && has_output) {
        return "SOURCE, SINK";
    }
    if (has_output) {
        return "SOURCE";
    }
    if (has_input) {
        return "SINK";
    }

    return "none";
}

[[nodiscard]] std::string plugin_author(const leakflow::PluginDescriptor& plugin)
{
    if (!plugin.author.empty()) {
        return plugin.author;
    }

    return plugin.owner;
}

[[nodiscard]] std::string plugin_license(const leakflow::PluginDescriptor& plugin)
{
    if (!plugin.license.empty()) {
        return plugin.license;
    }

    return "unknown";
}

[[nodiscard]] std::string plugin_version(const leakflow::PluginDescriptor& plugin)
{
    if (!plugin.version.empty()) {
        return plugin.version;
    }

    return "0.10";
}

[[nodiscard]] std::string plugin_filename(const leakflow::PluginDescriptor& plugin)
{
    return "lib" + plugin.name + ".so";
}

[[nodiscard]] std::string join_strings(const std::vector<std::string>& values)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << values[index];
    }
    return output.str();
}

[[nodiscard]] std::string direction_prefix(const std::optional<leakflow::PadDirection>& direction)
{
    if (!direction) {
        return {};
    }

    switch (*direction) {
    case leakflow::PadDirection::Input:
        return "Input ";
    case leakflow::PadDirection::Output:
        return "Output ";
    }

    return {};
}

[[nodiscard]] std::string metadata_target_single_label(const leakflow::MetadataPadTarget& target)
{
    const auto prefix = direction_prefix(target.direction);
    switch (target.kind) {
    case leakflow::MetadataPadTargetKind::AllPads:
        if (!target.direction) {
            return "All pads";
        }
        return *target.direction == leakflow::PadDirection::Input ? "All input pads" : "All output pads";
    case leakflow::MetadataPadTargetKind::PadName:
        return prefix + "pad: " + target.name;
    case leakflow::MetadataPadTargetKind::PadTemplate:
        return prefix + "pad template: " + target.name;
    }

    return "Metadata target";
}

[[nodiscard]] std::string metadata_target_label(const std::vector<leakflow::MetadataPadTarget>& targets)
{
    if (targets.empty()) {
        return "All pads";
    }
    if (targets.size() == 1) {
        return metadata_target_single_label(targets.front());
    }

    const auto same_kind = std::all_of(targets.begin(), targets.end(), [&](const auto& target) {
        return target.kind == targets.front().kind && target.direction == targets.front().direction;
    });
    if (same_kind && targets.front().kind == leakflow::MetadataPadTargetKind::PadName) {
        std::vector<std::string> names;
        names.reserve(targets.size());
        for (const auto& target : targets) {
            names.push_back(target.name);
        }
        return direction_prefix(targets.front().direction)
            + std::string("pads: ")
            + join_strings(names);
    }

    std::vector<std::string> labels;
    labels.reserve(targets.size());
    for (const auto& target : targets) {
        labels.push_back(metadata_target_single_label(target));
    }
    return "Targets: " + join_strings(labels);
}

void print_label(std::string_view label, const RenderContext& context)
{
    std::cout << styled(context, render::StyleRole::Label, label);
}

void print_heading(std::string_view heading, const RenderContext& context)
{
    std::cout << styled(context, render::StyleRole::Heading, heading) << '\n';
}

void print_key_value(std::string_view key, std::string_view value, const RenderContext& context)
{
    std::cout << "  ";
    print_label(key, context);
    std::cout << styled(context, render::StyleRole::Value, value) << '\n';
}

void print_pad_template(const leakflow::Pad& pad, const RenderContext& context)
{
    std::cout << "  "
              << styled(context, render::StyleRole::Label, pad_direction_name(pad.direction()) + " template")
              << ": '" << pad.name() << "'\n";
    std::cout << "    " << styled(context, render::StyleRole::Label, "Availability")
              << ": " << pad_presence_name(pad.presence()) << '\n';
    std::cout << "    " << styled(context, render::StyleRole::Label, "Capabilities") << ":\n";
    std::cout << "      " << styled(context, render::StyleRole::Feature, pad.caps().to_string()) << '\n';
}

void print_pad(const leakflow::Pad& pad, const RenderContext& context)
{
    std::cout << "  " << styled(context, render::StyleRole::Label, pad_direction_name(pad.direction()))
              << ": '" << pad.name() << "'\n";
    std::cout << "    " << styled(context, render::StyleRole::Label, "Pad Template")
              << ": '" << pad.name() << "'\n";
}

void print_property(const leakflow::PropertySpec& property, const RenderContext& context)
{
    std::cout << "  ";
    print_label(property.name, context);
    std::cout << " : " << property.description << '\n';
    std::cout << "    " << styled(context, render::StyleRole::Section, "flags") << ": "
              << styled(context, render::StyleRole::Value, "readable");
    if (property.writable) {
        std::cout << ", " << styled(context, render::StyleRole::Value, "writable");
    }
    std::cout << '\n';
    auto type_text = leakflow::property_value_type_name(property.default_value);
    if (property.optional) {
        type_text += " (optional)";
    }
    std::cout << "    " << styled(context, render::StyleRole::Section, "type") << ": "
              << styled(context, render::StyleRole::Type, type_text) << '\n';
    const auto default_text =
        property.optional ? std::string("null") : leakflow::property_value_to_string(property.default_value);
    std::cout << "    " << styled(context, render::StyleRole::Section, "Default") << ": "
              << styled(context, render::StyleRole::Value, default_text) << '\n';
    if (!property.unit.empty()) {
        std::cout << "    " << styled(context, render::StyleRole::Section, "Unit") << ": "
                  << styled(context, render::StyleRole::Value, property.unit) << '\n';
    }
    if (const auto* constraint = std::get_if<leakflow::StringEnumConstraint>(&property.constraint)) {
        std::cout << "    " << styled(context, render::StyleRole::Section, "Allowed values") << ": "
                  << styled(context, render::StyleRole::Value, join_strings(constraint->allowed_values)) << '\n';
    }
    if (!property.value_hint.empty()) {
        std::cout << "    " << styled(context, render::StyleRole::Section, "Accepted values") << ": "
                  << styled(context, render::StyleRole::Value, property.value_hint) << '\n';
    }
    {
        std::string effect = std::string(leakflow::property_effect_kind_name(property.effect.kind));
        effect += " (scope: ";
        effect += leakflow::property_invalidation_scope_name(property.effect.scope);
        if (!property.effect.output_pads.empty()) {
            effect += "; pads: ";
            effect += join_strings(property.effect.output_pads);
        }
        effect += ")";
        std::cout << "    " << styled(context, render::StyleRole::Section, "Effect") << ": "
                  << styled(context, render::StyleRole::Value, effect) << '\n';
    }
}

void print_telemetry(const leakflow::TelemetrySpec& telemetry, const RenderContext& context)
{
    const auto category = telemetry.kind == leakflow::TelemetryKind::Duration ? "profiling" : "monitoring";
    std::cout << "  ";
    print_label(telemetry.name, context);
    std::cout << " : " << telemetry.description << '\n';
    std::cout << "    " << styled(context, render::StyleRole::Section, "flags") << ": "
              << styled(context, render::StyleRole::Value, std::string("readable, runtime, ") + category) << '\n';
    std::cout << "    " << styled(context, render::StyleRole::Section, "type") << ": "
              << styled(context, render::StyleRole::Type, leakflow::property_value_type_name(telemetry.value_type))
              << '\n';
    if (!telemetry.unit.empty()) {
        std::cout << "    " << styled(context, render::StyleRole::Section, "Unit") << ": "
                  << styled(context, render::StyleRole::Value, telemetry.unit) << '\n';
    }
    if (!telemetry.value_hint.empty()) {
        std::cout << "    " << styled(context, render::StyleRole::Section, "Values") << ": "
                  << styled(context, render::StyleRole::Value, telemetry.value_hint) << '\n';
    }
}

void print_metadata_entry(const leakflow::ElementMetadataDescriptor& metadata,
    const RenderContext& context,
    std::string_view indent = "    ")
{
    std::cout << indent;
    print_label(metadata.key, context);
    std::cout << " : " << metadata.description << '\n';
    std::cout << indent << "  " << styled(context, render::StyleRole::Section, "type") << ": "
              << styled(context, render::StyleRole::Type, leakflow::property_value_type_name(metadata.value_type))
              << '\n';
    if (!metadata.example_values.empty()) {
        std::cout << indent << "  " << styled(context, render::StyleRole::Section, "Examples") << ": "
                  << styled(context, render::StyleRole::Value, join_strings(metadata.example_values)) << '\n';
    }
    if (!metadata.value_hint.empty()) {
        std::cout << indent << "  " << styled(context, render::StyleRole::Section, "Accepted values") << ": "
                  << styled(context, render::StyleRole::Value, metadata.value_hint) << '\n';
    }
    if (!metadata.details.empty()) {
        std::cout << indent << "  " << styled(context, render::StyleRole::Section, "Details") << ": "
                  << metadata.details << '\n';
    }
}

void print_metadata_subsection(std::string_view title,
    const std::vector<leakflow::ElementMetadataDescriptor>& metadata,
    const RenderContext& context)
{
    std::cout << "  " << styled(context, render::StyleRole::Label, title) << ":\n";
    if (metadata.empty()) {
        std::cout << "    none\n";
        return;
    }

    std::map<std::string, std::vector<const leakflow::ElementMetadataDescriptor*>> grouped;
    for (const auto& entry : metadata) {
        grouped[metadata_target_label(entry.pad_targets)].push_back(&entry);
    }

    for (const auto& [target, entries] : grouped) {
        std::cout << "    " << styled(context, render::StyleRole::Section, target) << ":\n";
        for (const auto* entry : entries) {
            print_metadata_entry(*entry, context, "      ");
        }
    }
}

void print_plugin_list(const RenderContext& context)
{
    for (const auto& plugin : descriptor_registry().plugins()) {
        for (const auto& element : plugin.elements) {
            std::cout << styled(context, render::StyleRole::Label, plugin.name)
                      << ":  " << styled(context, render::StyleRole::Feature, element.type_name)
                      << ": " << element.purpose << '\n';
        }
    }
}

void print_plugin_details(const leakflow::PluginDescriptor& plugin, const RenderContext& context)
{
    print_heading("Plugin Details:", context);
    print_key_value("Name                     ", plugin.name, context);
    print_key_value("Description              ", plugin.purpose, context);
    print_key_value("Filename                 ", plugin_filename(plugin), context);
    print_key_value("Version                  ", plugin_version(plugin), context);
    print_key_value("License                  ", plugin_license(plugin), context);
    print_key_value("Source module            ", "LeakFlow", context);
    print_key_value("Author                   ", plugin_author(plugin), context);

    std::cout << '\n';

    for (const auto& element : plugin.elements) {
        std::cout << "  " << styled(context, render::StyleRole::Feature, element.type_name)
                  << ": " << element.purpose << '\n';
    }

    std::cout << '\n';
    std::cout << "  " << plugin.elements.size() << " features:\n";
    std::cout << "  " << styled(context, render::StyleRole::Tree, context.glyphs.leaf)
              << " " << plugin.elements.size() << " elements\n";
}

struct ElementLookup {
    const leakflow::PluginDescriptor* plugin = nullptr;
    const leakflow::ElementDescriptor* element = nullptr;
};

[[nodiscard]] std::optional<ElementLookup> find_element(std::string_view name)
{
    if (const auto found = descriptor_registry().find_element(name)) {
        return ElementLookup{found->plugin, found->element};
    }

    return std::nullopt;
}

void print_plugin_details_for_element(const leakflow::PluginDescriptor& plugin, const RenderContext& context)
{
    print_heading("Plugin Details:", context);
    print_key_value("Name                     ", plugin.name, context);
    print_key_value("Description              ", plugin.purpose, context);
    print_key_value("Filename                 ", plugin_filename(plugin), context);
    print_key_value("Version                  ", plugin_version(plugin), context);
    print_key_value("License                  ", plugin_license(plugin), context);
    print_key_value("Source module            ", "LeakFlow", context);
    print_key_value("Author                   ", plugin_author(plugin), context);
}

void print_element_hierarchy(const leakflow::ElementDescriptor& element, const RenderContext& context)
{
    std::cout << styled(context, render::StyleRole::Type, "LeakFlowObject") << '\n';
    std::cout << styled(context, render::StyleRole::Tree, context.glyphs.leaf)
              << " "
              << styled(context, render::StyleRole::Type, "Element") << '\n';
    std::cout << "   "
              << styled(context, render::StyleRole::Tree, context.glyphs.leaf)
              << " "
              << styled(context, render::StyleRole::Type, element.type_name) << "\n\n";
}

void print_element_details(const leakflow::PluginDescriptor& plugin,
    const leakflow::ElementDescriptor& element,
    const RenderContext& context)
{
    print_heading("Factory Details:", context);
    print_key_value("Rank                     ", element_rank_text(element.rank), context);
    print_key_value("Long-name                ", element_long_name(element), context);
    print_key_value("Klass                    ", element_klass(element), context);
    print_key_value("Description              ", element.purpose, context);
    print_key_value("Author                   ", plugin_author(plugin), context);
    print_key_value("Replay-safe              ", element.can_replay ? "yes" : "no", context);

    std::cout << '\n';
    print_plugin_details_for_element(plugin, context);

    std::cout << '\n';
    print_element_hierarchy(element, context);

    print_heading("Element Flags:", context);
    std::cout << "  " << styled(context, render::StyleRole::Tree, context.glyphs.bullet)
              << " " << styled(context, render::StyleRole::Type, element_flags(element)) << "\n\n";

    print_heading("Pad Templates:", context);
    const auto has_explicit_templates = !element.pad_templates.empty();
    if (!has_explicit_templates && element.input_pads.empty() && element.output_pads.empty()) {
        std::cout << "  none\n";
    }
    if (has_explicit_templates) {
        for (const auto& pad_template : element.pad_templates) {
            print_pad_template(pad_template, context);
        }
    } else {
        for (const auto& pad : element.input_pads) {
            print_pad_template(pad, context);
        }
        for (const auto& pad : element.output_pads) {
            print_pad_template(pad, context);
        }
    }

    std::cout << '\n';
    std::cout << "Element has no clocking capabilities.\n";
    std::cout << "Element has no URI handling capabilities.\n\n";

    print_heading("Pads:", context);
    if (element.input_pads.empty() && element.output_pads.empty()) {
        std::cout << "  none\n";
    }
    for (const auto& pad : element.input_pads) {
        print_pad(pad, context);
    }
    for (const auto& pad : element.output_pads) {
        print_pad(pad, context);
    }

    std::cout << '\n';
    print_heading("Element Properties:", context);
    if (element.property_specs.empty()) {
        std::cout << "  none\n";
    }
    for (const auto& property : element.property_specs) {
        std::cout << '\n';
        print_property(property, context);
    }

    std::cout << '\n';
    print_heading("Element Telemetry:", context);
    if (element.telemetry_specs.empty()) {
        std::cout << "  none\n";
    }
    for (const auto& telemetry : element.telemetry_specs) {
        std::cout << '\n';
        print_telemetry(telemetry, context);
    }

    std::cout << "\n\n";
    print_heading("Element Metadata:", context);
    print_metadata_subsection("Set by element", element.metadata_set_by_element, context);
    print_metadata_subsection("Suggested user metadata", element.metadata_suggestions, context);
}

[[nodiscard]] const leakflow::PluginDescriptor* find_plugin(std::string_view name)
{
    return descriptor_registry().find_plugin(name);
}

void print_usage()
{
    std::cout << "Usage: leakflow-ls [--no-colors] [--color] [--ascii] [--utf8] [--no-pager] [PLUGIN|ELEMENT]\n";
    std::cout << "Environment:\n";
    std::cout << "  LEAKFLOW_LS_LESS    Options passed to less when paging output (default: RXF)\n";
}

[[nodiscard]] int print_inspection_result(std::optional<std::string_view> inspect_name, const RenderContext& context)
{
    if (!inspect_name) {
        print_plugin_list(context);
        return 0;
    }

    if (const auto* plugin = find_plugin(*inspect_name)) {
        print_plugin_details(*plugin, context);
        return 0;
    }

    if (const auto element = find_element(*inspect_name)) {
        print_element_details(*element->plugin, *element->element, context);
        return 0;
    }

    std::cerr << "unknown plugin or element: " << *inspect_name << '\n';
    print_usage();
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    auto color_mode = render::ColorMode::Auto;
    auto glyph_mode = render::GlyphMode::Utf8;
    auto pager_enabled = true;
    std::optional<std::string_view> inspect_name;

    for (const auto argument : std::span(argv + 1, static_cast<std::size_t>(argc - 1))) {
        const std::string_view value(argument);
        if (value == "--no-colors") {
            color_mode = render::ColorMode::Never;
            continue;
        }
        if (value == "--color" || value == "--colors" || value == "-C") {
            color_mode = render::ColorMode::Always;
            continue;
        }
        if (value == "--ascii") {
            glyph_mode = render::GlyphMode::Ascii;
            continue;
        }
        if (value == "--utf8" || value == "--unicode") {
            glyph_mode = render::GlyphMode::Utf8;
            continue;
        }
        if (value == "--no-pager") {
            pager_enabled = false;
            continue;
        }
        if (value == "--help" || value == "-h") {
            print_usage();
            return 0;
        }
        if (inspect_name) {
            print_usage();
            return 1;
        }
        inspect_name = value;
    }

    const RenderContext context{
        .styler = render::TerminalStyler(color_mode),
        .glyphs = render::glyphs_for(glyph_mode),
    };

    auto capture = CoutCapture{};
    const auto result = print_inspection_result(inspect_name, context);
    const auto output = capture.release();
    leakflow::apps::present_with_optional_pager(output, {.enabled = pager_enabled && result == 0});
    return result;
}
