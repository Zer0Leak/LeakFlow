#include "leakflow/plugins/core/buffer_file_src.hpp"

#include "buffer_file_manifest.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace leakflow::plugins::core {
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
            PropertySpec("path", std::string(), "directory to read the buffer from (a .lfbuf folder)"),
        },
        .keywords = {"buffer", "file", "source", "load", "deserialize"},
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
    const std::filesystem::path dir(path);

    const auto manifest = read_manifest(dir / "manifest.txt");
    const auto* codec = codecs_->find(manifest.payload_type);
    if (codec == nullptr) {
        throw std::invalid_argument("BufferFileSrc has no payload codec registered for '" + manifest.payload_type + "'");
    }

    auto payload = codec->load(dir);
    if (!payload) {
        throw std::runtime_error("BufferFileSrc payload codec returned no payload for '" + manifest.payload_type + "'");
    }

    Buffer buffer(manifest.caps);
    for (const auto& [key, value] : manifest.metadata) {
        buffer.set_metadata(key, value);
    }
    buffer.set_payload(std::move(payload));

    auto record = make_log_record(log::LogLevel::Debug, "element", "loaded buffer");
    record.fields.emplace("path", path);
    record.fields.emplace("payload.type", manifest.payload_type);
    record.fields.emplace("caps", buffer.caps().to_string());
    leakflow::log::write(std::move(record));

    return buffer;
}

} // namespace leakflow::plugins::core
