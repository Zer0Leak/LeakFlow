#include "leakflow/plugins/extras/buffer_file_sink.hpp"

#include "leakflow/extras/hdf5_buffer_archive.hpp"

#include <filesystem>
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

ElementDescriptor BufferFileSink::descriptor()
{
    return {
        .type_name = "BufferFileSink",
        .klass = "Sink/File/Buffer",
        .purpose = "persist a whole Buffer (caps + metadata + payload) to a single HDF5 file",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(any_caps_type)),
        },
        .output_pads = {},
        .property_specs = {
            PropertySpec("path", std::string(), "HDF5 file to write the buffer into (a .h5 file)"),
        },
        .keywords = {"buffer", "file", "sink", "save", "serialize", "hdf5", "h5"},
    };
}

BufferFileSink::BufferFileSink(std::shared_ptr<const PayloadCodecRegistry> codecs, std::string name)
    : Element(std::move(name))
    , codecs_(std::move(codecs))
{
    configure_from_descriptor(descriptor());
    if (!codecs_) {
        throw std::invalid_argument("BufferFileSink requires a PayloadCodecRegistry");
    }
}

std::optional<Buffer> BufferFileSink::process(std::optional<Buffer> input)
{
    if (!input) {
        return std::nullopt;
    }

    const auto payload = input->payload();
    if (!payload) {
        throw std::invalid_argument("BufferFileSink requires a buffer with a payload");
    }
    const auto& payload_type = payload->type_name();
    const auto* codec = codecs_->find(payload_type);
    if (codec == nullptr) {
        throw std::invalid_argument("BufferFileSink has no payload codec registered for '" + payload_type + "'");
    }

    const auto path = string_property_or(*this, "path", "");
    if (path.empty()) {
        throw std::invalid_argument("BufferFileSink path property must not be empty");
    }

    const std::filesystem::path file(path);
    if (file.has_parent_path()) {
        std::filesystem::create_directories(file.parent_path());
    }

    leakflow::extras::Hdf5BufferArchiveWriter archive(file);
    archive.set_caps_type(input->caps().type());
    for (const auto& [key, value] : input->caps().params()) {
        archive.set_caps_param(key, value);
    }
    for (const auto& [key, value] : input->metadata()) {
        archive.set_metadata(key, value);
    }
    // Persist the typed semantic axes as authoritative envelope identity (omitted when
    // the axis is none).
    if (!input->units().empty()) {
        archive.set_units(input->units().format());
    }
    if (!input->channels().empty()) {
        archive.set_channels(input->channels().format());
    }
    archive.set_payload_type(payload_type);
    codec->save(*payload, archive);

    auto record = make_log_record(log::LogLevel::Debug, "element", "wrote buffer");
    record.fields.emplace("path", path);
    record.fields.emplace("payload.type", payload_type);
    record.fields.emplace("caps", input->caps().to_string());
    leakflow::log::write(std::move(record));

    return std::nullopt;
}

} // namespace leakflow::plugins::extras
