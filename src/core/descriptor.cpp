#include "leakflow/core/descriptor.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace leakflow {

namespace {

[[nodiscard]] std::optional<std::size_t> find_unsigned_pad_placeholder(std::string_view text)
{
    const auto lower = text.find("%u");
    const auto upper = text.find("%U");
    if (lower == std::string_view::npos) {
        return upper == std::string_view::npos ? std::nullopt : std::optional<std::size_t>(upper);
    }
    if (upper == std::string_view::npos) {
        return lower;
    }

    return std::min(lower, upper);
}

void validate_metadata_pad_target(const MetadataPadTarget& target)
{
    switch (target.kind) {
    case MetadataPadTargetKind::AllPads:
        if (!target.name.empty()) {
            throw std::invalid_argument("all-pads metadata target cannot have a pad name");
        }
        return;
    case MetadataPadTargetKind::PadName:
        if (target.name.empty()) {
            throw std::invalid_argument("metadata pad target name cannot be empty");
        }
        return;
    case MetadataPadTargetKind::PadTemplate:
        if (target.name.empty()) {
            throw std::invalid_argument("metadata pad template target name cannot be empty");
        }
        if (!find_unsigned_pad_placeholder(target.name)) {
            throw std::invalid_argument("metadata pad template target must contain %u");
        }
        return;
    }
}

void validate_metadata_descriptor(const ElementMetadataDescriptor& metadata)
{
    if (metadata.key.empty()) {
        throw std::invalid_argument("metadata key cannot be empty");
    }
    if (metadata.description.empty()) {
        throw std::invalid_argument("metadata description cannot be empty");
    }
    for (const auto& target : metadata.pad_targets) {
        validate_metadata_pad_target(target);
    }
}

[[nodiscard]] bool has_property_named(const std::vector<PropertySpec>& specs, std::string_view name)
{
    for (const auto& spec : specs) {
        if (spec.name == name) {
            return true;
        }
    }

    return false;
}

} // namespace

ElementMetadataDescriptor make_element_metadata_descriptor(std::string key,
    PropertyValue value_type,
    std::string description,
    std::vector<std::string> example_values,
    std::string details,
    std::string value_hint,
    std::vector<MetadataPadTarget> pad_targets)
{
    ElementMetadataDescriptor metadata{
        .key = std::move(key),
        .value_type = std::move(value_type),
        .description = std::move(description),
        .example_values = std::move(example_values),
        .details = std::move(details),
        .value_hint = std::move(value_hint),
        .pad_targets = std::move(pad_targets),
    };
    validate_metadata_descriptor(metadata);
    return metadata;
}

std::string element_default_name_prefix(std::string_view type_name)
{
    std::string prefix;
    prefix.reserve(type_name.size());
    for (const auto character : type_name) {
        if (std::isalnum(static_cast<unsigned char>(character)) == 0) {
            continue;
        }
        prefix.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }

    if (prefix.empty()) {
        throw std::invalid_argument("element type name cannot produce a default instance name");
    }

    return prefix;
}

PropertySpec element_name_property_spec(std::string default_name)
{
    if (default_name.empty()) {
        throw std::invalid_argument("element name property default cannot be empty");
    }

    return PropertySpec(
        "name",
        std::move(default_name),
        "The name of the element instance");
}

ElementDescriptor with_common_element_properties(ElementDescriptor descriptor)
{
    if (!has_property_named(descriptor.property_specs, "name")) {
        descriptor.property_specs.insert(
            descriptor.property_specs.begin(),
            element_name_property_spec(element_default_name_prefix(descriptor.type_name) + "0"));
    }

    return descriptor;
}

PluginDescriptor with_common_element_properties(PluginDescriptor descriptor)
{
    for (auto& element : descriptor.elements) {
        element = with_common_element_properties(std::move(element));
    }

    return descriptor;
}

MetadataPadTarget metadata_all_pads()
{
    return {};
}

MetadataPadTarget metadata_all_pads(PadDirection direction)
{
    return {
        .kind = MetadataPadTargetKind::AllPads,
        .direction = direction,
    };
}

MetadataPadTarget metadata_pad(std::string name)
{
    MetadataPadTarget target{
        .kind = MetadataPadTargetKind::PadName,
        .name = std::move(name),
    };
    validate_metadata_pad_target(target);
    return target;
}

MetadataPadTarget metadata_pad(PadDirection direction, std::string name)
{
    MetadataPadTarget target{
        .kind = MetadataPadTargetKind::PadName,
        .direction = direction,
        .name = std::move(name),
    };
    validate_metadata_pad_target(target);
    return target;
}

MetadataPadTarget metadata_pad_template(std::string name)
{
    MetadataPadTarget target{
        .kind = MetadataPadTargetKind::PadTemplate,
        .name = std::move(name),
    };
    validate_metadata_pad_target(target);
    return target;
}

MetadataPadTarget metadata_pad_template(PadDirection direction, std::string name)
{
    MetadataPadTarget target{
        .kind = MetadataPadTargetKind::PadTemplate,
        .direction = direction,
        .name = std::move(name),
    };
    validate_metadata_pad_target(target);
    return target;
}

bool metadata_pad_template_matches(std::string_view pad_template, std::string_view pad_name)
{
    const auto placeholder = find_unsigned_pad_placeholder(pad_template);
    if (!placeholder) {
        return false;
    }

    const auto prefix = pad_template.substr(0, *placeholder);
    const auto suffix = pad_template.substr(*placeholder + 2);
    if (!pad_name.starts_with(prefix) || !pad_name.ends_with(suffix)) {
        return false;
    }

    const auto digits = pad_name.substr(prefix.size(), pad_name.size() - prefix.size() - suffix.size());
    if (digits.empty()) {
        return false;
    }
    for (const auto character : digits) {
        if (std::isdigit(static_cast<unsigned char>(character)) == 0) {
            return false;
        }
    }

    return true;
}

ElementDescriptor& register_metadata_set_by_element(
    ElementDescriptor& descriptor,
    ElementMetadataDescriptor metadata)
{
    validate_metadata_descriptor(metadata);
    descriptor.metadata_set_by_element.push_back(std::move(metadata));
    return descriptor;
}

ElementDescriptor& register_metadata_suggestion(
    ElementDescriptor& descriptor,
    ElementMetadataDescriptor metadata)
{
    validate_metadata_descriptor(metadata);
    descriptor.metadata_suggestions.push_back(std::move(metadata));
    return descriptor;
}

} // namespace leakflow
