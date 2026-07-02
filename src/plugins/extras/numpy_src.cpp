#include "leakflow/plugins/extras/numpy_src.hpp"

#include "leakflow/extras/numpy_payload.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
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

ElementDescriptor NumpySrc::descriptor()
{
    return {
        .type_name = "NumpySrc",
        .klass = "Source/File/Numpy",
        .purpose = "load one NumPy .npy file as a NumpyPayload",
        .input_pads = {},
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(leakflow::extras::numpy_array_caps_type)),
        },
        .property_specs = {
            PropertySpec("path", std::string(), "path to read as a NumPy .npy array"),
        },
        .keywords = {"numpy", "npy", "source", "extras"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "routing.element",
                std::string(),
                "element instance name that produced the buffer",
                {"numpysrc0"}),
            make_element_metadata_descriptor(
                "origin.file.path",
                std::string(),
                "input NumPy file path",
                {"traces/aes/sync/aes_sync_poi/key_01/traces.npy"}),
            make_element_metadata_descriptor(
                "origin.file.size",
                std::int64_t{},
                "input file size in bytes",
                {"40000000"}),
        },
        .metadata_suggestions = {
            make_element_metadata_descriptor(
                "capture.source",
                std::string(),
                "capture hardware, simulator, or acquisition source",
                {"ChipWhisperer"}),
            make_element_metadata_descriptor(
                "capture.dataset.name",
                std::string(),
                "dataset or experiment identifier for downstream reporting",
                {"aes_sync_poi"}),
            make_element_metadata_descriptor(
                "origin.role",
                std::string(),
                "semantic role of the array, such as traces, plaintexts, keys, or labels",
                {"traces", "plaintexts", "keys", "labels"}),
            make_element_metadata_descriptor(
                "capture.sample_rate_hz",
                0.0,
                "trace sample rate for downstream plot time axes",
                {"29454545.454545453"},
                "TracePlot can use this metadata for time-axis rendering."),
        },
    };
}

NumpySrc::NumpySrc(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> NumpySrc::process(std::optional<Buffer> input)
{
    if (input) {
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "forwarded existing input"));
        return input;
    }

    const std::filesystem::path path = string_property_or(*this, "path", "");
    if (path.empty()) {
        throw std::invalid_argument("NumpySrc path property must not be empty");
    }

    auto payload = leakflow::extras::load_npy(path);

    Buffer buffer{payload.caps()};
    buffer.set_metadata("routing.element", name());
    buffer.set_metadata("origin.file.path", path.string());
    buffer.set_metadata("origin.file.size", std::to_string(std::filesystem::file_size(path)));
    buffer.set_payload(std::make_shared<leakflow::extras::NumpyPayload>(std::move(payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "loaded NumPy file");
    record.fields.emplace("caps", buffer.caps().to_string());
    record.fields.emplace("origin.file.path", path.string());
    record.fields.emplace("origin.file.size", buffer.metadata("origin.file.size"));
    leakflow::log::write(std::move(record));
    return buffer;
}

} // namespace leakflow::plugins::extras
