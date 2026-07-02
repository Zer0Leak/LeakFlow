#include "leakflow/plugins/core/file_src.hpp"

#include "leakflow/plugins/core/bytes_payload.hpp"
#include "core_plugin_constants.hpp"
#include "property_helpers.hpp"

#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace leakflow::plugins::core {
namespace {

[[nodiscard]] std::string read_file_bytes(const std::string& path)
{
    if (path.empty()) {
        throw std::invalid_argument("FileSrc path property must not be empty");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("FileSrc could not open input file");
    }

    std::ostringstream bytes;
    bytes << input.rdbuf();
    return bytes.str();
}

} // namespace

ElementDescriptor FileSrc::descriptor()
{
    return {
        .type_name = "FileSrc",
        .klass = "Source/File",
        .purpose = "read one file as raw byte payload",
        .input_pads = {},
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(core_pad_caps_type)),
        },
        .property_specs = {
            PropertySpec("path", std::string(), "path to read as raw bytes"),
            PropertySpec("caps_type", std::string("leakflow/bytes"), "caps type emitted by FileSrc"),
        },
        .keywords = {"file", "source", "bytes"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "origin.file.path",
                std::string(),
                "input file path",
                {"data/raw.bin"}),
            make_element_metadata_descriptor(
                "origin.file.size",
                std::int64_t{},
                "input file size in bytes",
                {"4096"}),
            make_element_metadata_descriptor(
                "routing.element",
                std::string(),
                "element instance name that produced the buffer",
                {"filesrc0"}),
        },
        .metadata_suggestions = {
            make_element_metadata_descriptor(
                "capture.dataset.name",
                std::string(),
                "dataset or experiment identifier for downstream reporting",
                {"aes_sync_poi"}),
            make_element_metadata_descriptor(
                "origin.role",
                std::string(),
                "semantic role of the raw bytes, such as traces or labels",
                {"traces", "labels"}),
            make_element_metadata_descriptor(
                "capture.source",
                std::string(),
                "capture hardware, simulator, or acquisition source",
                {"ChipWhisperer"}),
        },
    };
}

FileSrc::FileSrc(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> FileSrc::process(std::optional<Buffer> input)
{
    if (input) {
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "forwarded existing input"));
        return input;
    }

    const auto path = string_property_or(*this, "path", "");
    auto bytes = read_file_bytes(path);

    Buffer buffer(Caps(string_property_or(*this, "caps_type", "leakflow/bytes")));
    buffer.set_metadata("origin.file.path", path);
    buffer.set_metadata("origin.file.size", std::to_string(bytes.size()));
    buffer.set_metadata("routing.element", name());
    buffer.set_payload(std::make_shared<BytesPayload>(std::move(bytes)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "read raw byte file");
    record.fields.emplace("caps", buffer.caps().to_string());
    record.fields.emplace("origin.file.path", path);
    record.fields.emplace("origin.file.size", buffer.metadata("origin.file.size"));
    leakflow::log::write(std::move(record));
    return buffer;
}

} // namespace leakflow::plugins::core
