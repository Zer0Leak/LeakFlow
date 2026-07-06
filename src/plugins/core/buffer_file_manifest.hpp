#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/caps.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace leakflow::plugins::core {

// The human-readable envelope BufferFileSink writes and BufferFileSrc reads, one
// `key=value` per line. It carries everything except the payload body (which the
// registered PayloadCodec writes/reads separately):
//
//   caps.type=<type>
//   caps.param.<key>=<value>
//   meta.<key>=<value>
//   payload.type=<payload type_name>
//
// Values are escaped for backslash and newline so arbitrary metadata round-trips.
struct BufferManifest {
    Caps caps = Caps(any_caps_type);
    std::map<std::string, std::string> metadata;
    std::string payload_type;
};

[[nodiscard]] inline std::string manifest_escape(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (const char character : value) {
        if (character == '\\') {
            out += "\\\\";
        } else if (character == '\n') {
            out += "\\n";
        } else {
            out += character;
        }
    }
    return out;
}

[[nodiscard]] inline std::string manifest_unescape(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '\\' && index + 1 < value.size()) {
            const char next = value[++index];
            out += next == 'n' ? '\n' : next;
        } else {
            out += value[index];
        }
    }
    return out;
}

inline void write_manifest(const std::filesystem::path& file, const Buffer& buffer, const std::string& payload_type)
{
    std::ofstream out(file, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("BufferFileSink could not write manifest: " + file.string());
    }
    out << "caps.type=" << manifest_escape(buffer.caps().type()) << '\n';
    for (const auto& [key, value] : buffer.caps().params()) {
        out << "caps.param." << manifest_escape(key) << '=' << manifest_escape(value) << '\n';
    }
    for (const auto& [key, value] : buffer.metadata()) {
        out << "meta." << manifest_escape(key) << '=' << manifest_escape(value) << '\n';
    }
    out << "payload.type=" << manifest_escape(payload_type) << '\n';
    if (!out) {
        throw std::runtime_error("BufferFileSink failed while writing manifest: " + file.string());
    }
}

[[nodiscard]] inline BufferManifest read_manifest(const std::filesystem::path& file)
{
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("BufferFileSrc could not read manifest: " + file.string());
    }

    std::string caps_type;
    Caps::Params caps_params;
    BufferManifest manifest;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            continue;
        }
        const auto key = line.substr(0, separator);
        const auto value = manifest_unescape(line.substr(separator + 1));
        if (key == "caps.type") {
            caps_type = value;
        } else if (key.rfind("caps.param.", 0) == 0) {
            caps_params.emplace(manifest_unescape(key.substr(std::string("caps.param.").size())), value);
        } else if (key.rfind("meta.", 0) == 0) {
            manifest.metadata.emplace(manifest_unescape(key.substr(std::string("meta.").size())), value);
        } else if (key == "payload.type") {
            manifest.payload_type = value;
        }
    }

    if (caps_type.empty() || manifest.payload_type.empty()) {
        throw std::invalid_argument("BufferFileSrc manifest is missing caps.type or payload.type: " + file.string());
    }
    manifest.caps = Caps(std::move(caps_type), std::move(caps_params));
    return manifest;
}

} // namespace leakflow::plugins::core
