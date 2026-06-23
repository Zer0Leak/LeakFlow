#include "leakflow/plugins/core/fake_src.hpp"

#include "core_plugin_constants.hpp"
#include "property_helpers.hpp"

#include <utility>

namespace leakflow::plugins::core {

ElementDescriptor FakeSrc::descriptor()
{
    return {
        .type_name = "FakeSrc",
        .klass = "Source/Fake",
        .purpose = "produce simple test buffers",
        .input_pads = {},
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(core_pad_caps_type)),
        },
        .property_specs = {
            PropertySpec("caps_type", std::string("sca/fake"), "caps type emitted by FakeSrc"),
            PropertySpec("metadata_key", std::string("source"), "metadata key emitted by FakeSrc"),
            PropertySpec("metadata_value", std::string("fake"), "metadata value emitted by FakeSrc"),
        },
        .keywords = {"fake", "source", "test"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "<metadata_key>",
                std::string(),
                "metadata key configured by the metadata_key property",
                {"source=fake"}),
            make_element_metadata_descriptor(
                "element",
                std::string(),
                "element instance name that produced the buffer",
                {"fakesrc0"}),
        },
    };
}

FakeSrc::FakeSrc(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> FakeSrc::process(std::optional<Buffer> input)
{
    if (input) {
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "forwarded existing input"));
        return input;
    }

    Buffer buffer(Caps(string_property_or(*this, "caps_type", "sca/fake")));
    buffer.set_metadata(
        string_property_or(*this, "metadata_key", "source"),
        string_property_or(*this, "metadata_value", "fake"));
    buffer.set_metadata("element", name());

    auto record = make_log_record(log::LogLevel::Debug, "element", "produced fake buffer");
    record.fields.emplace("caps", buffer.caps().to_string());
    leakflow::log::write(std::move(record));
    return buffer;
}

} // namespace leakflow::plugins::core
