#include "leakflow/core/descriptor.hpp"
#include "leakflow/core/descriptor_registry.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <variant>

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
    leakflow::ElementDescriptor element_descriptor{
        .type_name = "test/PoiFinder",
        .purpose = "find points of interest",
        .pad_templates = {
            leakflow::Pad(
                "poi_%u",
                leakflow::PadDirection::Output,
                leakflow::Caps(leakflow::any_caps_type),
                leakflow::PadPresence::OnRequest),
        },
        .input_pads = {
            leakflow::Pad("traces", leakflow::PadDirection::Input, leakflow::Caps("sca/traces")),
        },
        .output_pads = {
            leakflow::Pad("pois", leakflow::PadDirection::Output, leakflow::Caps("sca/pois")),
        },
        .property_specs = {
            leakflow::PropertySpec(
                "poi_count",
                std::int64_t{20},
                "number of PoIs to emit",
                "count",
                leakflow::IntRangeConstraint{1, 10000}),
            leakflow::PropertySpec(
                "method",
                std::string("pearson"),
                "correlation method",
                "",
                leakflow::StringEnumConstraint{{"pearson"}}),
        },
        .telemetry_specs = {
            leakflow::TelemetrySpec(
                "processed",
                std::int64_t{0},
                "number of processed buffers",
                "buffers"),
        },
        .keywords = {"poi", "correlation"},
        .metadata_set_by_element = {
            leakflow::make_element_metadata_descriptor(
                "poi.method",
                std::string(),
                "PoI selection method",
                {"pearson"},
                "Used by downstream reports.",
                "pearson",
                {leakflow::metadata_pad(leakflow::PadDirection::Output, "pois")}),
        },
        .metadata_suggestions = {
            leakflow::make_element_metadata_descriptor(
                "dataset.name",
                std::string(),
                "dataset identifier useful in reports",
                {"aes_sync_poi"},
                {},
                {},
                {leakflow::metadata_pad(leakflow::PadDirection::Input, "traces")}),
        },
    };

    leakflow::register_metadata_set_by_element(
        element_descriptor,
        leakflow::make_element_metadata_descriptor(
            "poi.count",
            std::int64_t{},
            "number of points of interest emitted",
            {"20"},
            {},
            {},
            {leakflow::metadata_pad_template(leakflow::PadDirection::Output, "poi_%u")}));
    leakflow::register_metadata_suggestion(
        element_descriptor,
        leakflow::make_element_metadata_descriptor(
            "capture.source",
            std::string(),
            "capture source",
            {"ChipWhisperer"}));

    if (!expect(element_descriptor.type_name == "test/PoiFinder", "element descriptor type name was not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.purpose == "find points of interest", "element descriptor purpose was not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.pad_templates.size() == 1,
            "element descriptor pad templates were not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.pad_templates[0].name() == "poi_%u",
            "element descriptor pad template name was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.pad_templates[0].presence() == leakflow::PadPresence::OnRequest,
            "element descriptor pad template presence was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.input_pads.size() == 1, "element descriptor input pads were not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.input_pads[0].name() == "traces", "element descriptor input pad name was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.output_pads.size() == 1, "element descriptor output pads were not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.output_pads[0].caps().type() == "sca/pois",
            "element descriptor output pad caps were wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.property_specs.size() == 2, "element descriptor property specs were not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.property_specs[0].name == "poi_count",
            "element descriptor property spec name was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.telemetry_specs.size() == 1,
            "element descriptor telemetry specs were not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.telemetry_specs[0].name == "processed" &&
                    leakflow::property_value_type_name(element_descriptor.telemetry_specs[0].value_type) == "integer",
            "element descriptor telemetry spec was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.keywords.size() == 2, "element descriptor keywords were not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.keywords[1] == "correlation", "element descriptor keyword value was wrong")) {
        return 1;
    }
    if (!expect(leakflow::element_default_name_prefix(element_descriptor.type_name) == "testpoifinder",
            "default element name prefix was wrong")) {
        return 1;
    }
    const auto common_element_descriptor = leakflow::with_common_element_properties(element_descriptor);
    if (!expect(common_element_descriptor.property_specs.size() == 3,
            "common element properties were not added")) {
        return 1;
    }
    if (!expect(common_element_descriptor.property_specs[0].name == "name",
            "common element name property was not first")) {
        return 1;
    }
    if (!expect(std::get<std::string>(common_element_descriptor.property_specs[0].default_value) == "testpoifinder0",
            "common element name property default was wrong")) {
        return 1;
    }
    if (!expect(common_element_descriptor.property_specs[1].name == "poi_count",
            "common element property insertion changed element-specific ordering")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element.size() == 2,
            "element descriptor set metadata was not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].key == "poi.method",
            "element descriptor set metadata key was wrong")) {
        return 1;
    }
    if (!expect(leakflow::property_value_type_name(element_descriptor.metadata_set_by_element[0].value_type) == "string",
            "element descriptor set metadata value type was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].example_values.size() == 1
                && element_descriptor.metadata_set_by_element[0].example_values[0] == "pearson",
            "element descriptor set metadata example was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].details == "Used by downstream reports.",
            "element descriptor set metadata details were wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].value_hint == "pearson",
            "element descriptor set metadata hint was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[1].key == "poi.count",
            "element descriptor registered set metadata key was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].pad_targets.size() == 1,
            "element descriptor metadata pad target was not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].pad_targets[0].kind
                == leakflow::MetadataPadTargetKind::PadName,
            "element descriptor metadata pad target kind was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].pad_targets[0].direction
                == leakflow::PadDirection::Output,
            "element descriptor metadata pad target direction was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[0].pad_targets[0].name == "pois",
            "element descriptor metadata pad target name was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_set_by_element[1].pad_targets[0].kind
                == leakflow::MetadataPadTargetKind::PadTemplate,
            "element descriptor metadata template target kind was wrong")) {
        return 1;
    }
    if (!expect(leakflow::metadata_pad_template_matches("poi_%u", "poi_7"),
            "metadata pad template did not match unsigned pad name")) {
        return 1;
    }
    if (!expect(!leakflow::metadata_pad_template_matches("poi_%u", "poi_last"),
            "metadata pad template matched non-numeric pad name")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_suggestions.size() == 2,
            "element descriptor metadata suggestions were not stored")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_suggestions[0].key == "dataset.name",
            "element descriptor metadata suggestion key was wrong")) {
        return 1;
    }
    if (!expect(element_descriptor.metadata_suggestions[1].key == "capture.source",
            "element descriptor registered metadata suggestion key was wrong")) {
        return 1;
    }

    leakflow::PluginDescriptor plugin_descriptor{
        .name = "leakflow_plugins_test",
        .owner = "LeakFlow",
        .author = "Zer0Leak <edgard.lima@gmail.com>",
        .license = "Apache-2.0",
        .purpose = "test plugin descriptors",
        .keywords = {"test", "poi"},
        .elements = {element_descriptor},
    };

    if (!expect(plugin_descriptor.name == "leakflow_plugins_test", "plugin descriptor name was not stored")) {
        return 1;
    }
    if (!expect(plugin_descriptor.owner == "LeakFlow", "plugin descriptor owner was not stored")) {
        return 1;
    }
    if (!expect(plugin_descriptor.author == "Zer0Leak <edgard.lima@gmail.com>",
            "plugin descriptor author was not stored")) {
        return 1;
    }
    if (!expect(plugin_descriptor.license == "Apache-2.0", "plugin descriptor license was not stored")) {
        return 1;
    }
    if (!expect(plugin_descriptor.purpose == "test plugin descriptors", "plugin descriptor purpose was not stored")) {
        return 1;
    }
    if (!expect(plugin_descriptor.keywords.size() == 2, "plugin descriptor keywords were not stored")) {
        return 1;
    }
    if (!expect(plugin_descriptor.elements.size() == 1, "plugin descriptor elements were not stored")) {
        return 1;
    }
    if (!expect(plugin_descriptor.elements[0].type_name == "test/PoiFinder",
            "plugin descriptor element metadata was wrong")) {
        return 1;
    }

    leakflow::DescriptorRegistry registry;
    registry.register_plugin(plugin_descriptor);
    if (!expect(registry.plugins().size() == 1, "descriptor registry plugin count was wrong")) {
        return 1;
    }
    if (!expect(registry.find_plugin("leakflow_plugins_test") != nullptr,
            "descriptor registry did not find plugin")) {
        return 1;
    }
    const auto found_element = registry.find_element("testpoifinder");
    if (!expect(found_element.has_value(), "descriptor registry did not find normalized element")) {
        return 1;
    }
    if (!expect(found_element->plugin->name == "leakflow_plugins_test",
            "descriptor registry element plugin was wrong")) {
        return 1;
    }
    if (!expect(found_element->element->type_name == "test/PoiFinder",
            "descriptor registry element descriptor was wrong")) {
        return 1;
    }
    if (!expect(found_element->element->property_specs.size() == 3,
            "descriptor registry did not publish common element properties")) {
        return 1;
    }
    if (!expect(found_element->element->property_specs[0].name == "name",
            "descriptor registry did not publish the name property first")) {
        return 1;
    }
    // The common "process" duration channel is prepended to every element by
    // with_common_element_properties (like the common "name" property), so the
    // element-declared "processed" telemetry follows it.
    if (!expect(found_element->element->telemetry_specs.size() == 2,
            "descriptor registry did not publish telemetry specs")) {
        return 1;
    }
    if (!expect(found_element->element->telemetry_specs[0].name == "process" &&
                    found_element->element->telemetry_specs[0].kind == leakflow::TelemetryKind::Duration,
            "descriptor registry did not publish the common process timing channel first")) {
        return 1;
    }
    if (!expect(found_element->element->telemetry_specs[1].name == "processed",
            "descriptor registry did not publish the element telemetry spec")) {
        return 1;
    }

    return 0;
}
