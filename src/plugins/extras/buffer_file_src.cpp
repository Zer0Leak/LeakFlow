#include "leakflow/plugins/extras/buffer_file_src.hpp"

#include "leakflow/extras/hdf5_buffer_archive.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace leakflow::plugins::extras {
namespace {

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }
    return fallback;
}

} // namespace

ElementDescriptor BufferFileSrc::descriptor()
{
    return {
        .type_name = "BufferFileSrc",
        .long_name = "Buffer File Source",
        .rank = ElementRank::Primary,
        .klass = "Source/File/Buffer",
        .purpose = "reload a whole Buffer (caps + metadata + payload) written by BufferFileSink",
        .input_pads = {},
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(any_caps_type)),
        },
        .property_specs = {
            PropertySpec("path", std::string(), "HDF5 file to read the buffer from (a .h5 file)"),
        },
        .keywords = {"buffer", "file", "source", "load", "deserialize", "hdf5", "h5"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.layout",
                std::string(),
                "payload layout restored from the archived buffer metadata",
                {"axis_0/axis_1", "trace/sample", "byte"}),
        },
    };
}

BufferFileSrc::BufferFileSrc(std::shared_ptr<const PayloadCodecRegistry> codecs, std::string name)
    : Element(std::move(name))
    , codecs_(std::move(codecs))
{
    configure_from_descriptor(descriptor());
    if (!codecs_) {
        throw std::invalid_argument("BufferFileSrc requires a PayloadCodecRegistry");
    }
}

std::optional<Buffer> BufferFileSrc::process(std::optional<Buffer> input)
{
    if (input) {
        return input;
    }

    const auto path = string_property_or(*this, "path", "");
    if (path.empty()) {
        throw std::invalid_argument("BufferFileSrc path property must not be empty");
    }

    leakflow::extras::Hdf5BufferArchiveReader archive(path);
    const auto* codec = codecs_->find(archive.payload_type());
    if (codec == nullptr) {
        throw std::invalid_argument(
            "BufferFileSrc has no payload codec registered for '" + archive.payload_type() + "'");
    }

    auto payload = codec->load(archive);
    if (!payload) {
        throw std::runtime_error(
            "BufferFileSrc payload codec returned no payload for '" + archive.payload_type() + "'");
    }

    Buffer buffer(Caps(archive.caps_type(), archive.caps_params()));
    buffer.set_payload(std::move(payload));
    // Archive metadata is authoritative. In particular, restore a semantic
    // payload.layout after set_payload() has installed the payload's generic fallback.
    for (const auto& [key, value] : archive.metadata()) {
        buffer.set_metadata(key, value);
    }
    // Restore the typed semantic axes (empty / schema-1 files parse back to none).
    buffer.set_units(Units::parse(archive.units()));
    buffer.set_channels(Channels::parse(archive.channels()));

    auto record = make_log_record(log::LogLevel::Debug, "element", "loaded buffer");
    record.fields.emplace("path", path);
    record.fields.emplace("payload.type", archive.payload_type());
    record.fields.emplace("caps", buffer.caps().to_string());
    leakflow::log::write(std::move(record));

    return buffer;
}

} // namespace leakflow::plugins::extras
