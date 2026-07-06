#include "leakflow/plugins/core/file_sink.hpp"

#include "leakflow/plugins/core/buffer_summary.hpp"
#include "leakflow/plugins/core/bytes_payload.hpp"
#include "core_plugin_constants.hpp"
#include "property_helpers.hpp"

#include <fstream>
#include <ios>
#include <stdexcept>
#include <utility>

namespace leakflow::plugins::core {
namespace {

void write_file_bytes(const std::string& path, const std::string& bytes, bool append)
{
    if (path.empty()) {
        return;
    }

    std::ofstream output(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    if (!output) {
        throw std::runtime_error("FileSink could not open output file");
    }

    output << bytes;
}

} // namespace

ElementDescriptor FileSink::descriptor()
{
    return {
        .type_name = "FileSink",
        .klass = "Sink/File/Bytes",
        .purpose = "write raw byte payloads to a file",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(core_pad_caps_type)),
        },
        .output_pads = {},
        .property_specs = {
            PropertySpec("path", std::string(), "path to write raw bytes"),
            PropertySpec("append", false, "append to the output file instead of truncating it"),
        },
        .keywords = {"file", "sink", "bytes"},
    };
}

FileSink::FileSink(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> FileSink::process(std::optional<Buffer> input)
{
    received_ = input.has_value();
    last_buffer_ = input;
    last_bytes_.clear();

    if (input) {
        if (const auto payload = input->payload_as<BytesPayload>()) {
            last_bytes_ = payload->bytes();
        } else {
            last_bytes_ = summarize_buffer(*input, 1);
        }

        write_file_bytes(
            string_property_or(*this, "path", ""),
            last_bytes_,
            bool_property_or(*this, "append", false));

        auto record = make_log_record(log::LogLevel::Debug, "element", "wrote raw byte file");
        record.fields.emplace("bytes", std::to_string(last_bytes_.size()));
        record.fields.emplace("path", string_property_or(*this, "path", ""));
        leakflow::log::write(std::move(record));
    } else {
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "received no buffer"));
    }

    return std::nullopt;
}

bool FileSink::received() const
{
    return received_;
}

const std::optional<Buffer>& FileSink::last_buffer() const
{
    return last_buffer_;
}

const std::string& FileSink::last_bytes() const
{
    return last_bytes_;
}

} // namespace leakflow::plugins::core
