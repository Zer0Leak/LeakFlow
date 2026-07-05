#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plugins/base/base_elements.hpp"
#include "leakflow/plugins/base/descriptor_catalog.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

template <typename Exception, typename Function>
bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }

    return false;
}

std::filesystem::path fixture_path(const char* name)
{
    return std::filesystem::path(LEAKFLOW_TORCH_FIXTURE_DIR) / name;
}

std::filesystem::path temp_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

std::vector<char> read_binary_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

const leakflow::ElementMetadataDescriptor* find_metadata(
    const std::vector<leakflow::ElementMetadataDescriptor>& descriptors,
    std::string_view key)
{
    for (const auto& descriptor : descriptors) {
        if (descriptor.key == key) {
            return &descriptor;
        }
    }

    return nullptr;
}

const leakflow::PropertySpec* find_property(
    const std::vector<leakflow::PropertySpec>& specs,
    std::string_view name)
{
    for (const auto& spec : specs) {
        if (spec.name == name) {
            return &spec;
        }
    }

    return nullptr;
}

} // namespace

int main()
{
    namespace base_plugin = leakflow::plugins::base;

    const auto traces_path = fixture_path("traces_first_50.pt");
    const auto saved_path = temp_path("leakflow_torch_file_io_roundtrip.pt");
    std::filesystem::remove(saved_path);

    base_plugin::TorchFileSrc fixture_source;
    fixture_source.set_property("path", traces_path.string());
    auto torch_buffer = fixture_source.process(std::nullopt);
    if (!expect(torch_buffer.has_value(), "TorchFileSrc fixture load did not produce a buffer")) {
        return 1;
    }

    const auto saved_payload = torch_buffer->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(saved_payload != nullptr, "TorchFileSrc fixture buffer did not carry TorchTensorPayload")) {
        return 1;
    }

    base_plugin::TorchFileSink sink;
    sink.set_property("path", saved_path.string());
    auto sink_output = sink.process(torch_buffer);
    if (!expect(sink.received(), "TorchFileSink did not receive a buffer")) {
        return 1;
    }
    if (!expect(!sink_output.has_value(), "TorchFileSink should consume the buffer")) {
        return 1;
    }
    if (!expect(std::filesystem::exists(saved_path), "TorchFileSink did not create the .pt file")) {
        return 1;
    }

    const auto pickle_value = torch::pickle_load(read_binary_file(saved_path));
    if (!expect(pickle_value.isTensor(), "TorchFileSink did not write a pickle-compatible tensor")) {
        return 1;
    }
    if (!expect(torch::equal(pickle_value.toTensor(), saved_payload->tensor()),
                "pickle-compatible tensor values changed")) {
        return 1;
    }

    base_plugin::TorchFileSrc torch_source;
    torch_source.set_property("path", saved_path.string());
    auto loaded_buffer = torch_source.process(std::nullopt);
    if (!expect(loaded_buffer.has_value(), "TorchFileSrc did not produce a buffer")) {
        return 1;
    }
    if (!expect(loaded_buffer->caps().type() == leakflow::base::torch_tensor_caps_type,
                "TorchFileSrc emitted wrong caps type")) {
        return 1;
    }
    if (!expect(loaded_buffer->caps().param(leakflow::base::caps_param_dtype) == "float32",
                "TorchFileSrc emitted wrong dtype caps")) {
        return 1;
    }
    if (!expect(loaded_buffer->caps().param(leakflow::base::caps_param_device) == "cpu",
                "TorchFileSrc emitted wrong device caps")) {
        return 1;
    }
    if (!expect(loaded_buffer->caps().param(leakflow::base::caps_param_rank) == "2",
                "TorchFileSrc emitted wrong rank caps")) {
        return 1;
    }
    if (!expect(loaded_buffer->caps().param(leakflow::base::caps_param_shape) == "[50,5000]",
                "TorchFileSrc emitted wrong shape caps")) {
        return 1;
    }
    if (!expect(!loaded_buffer->has_metadata("routing.element"), "TorchFileSrc still stamped element metadata")) {
        return 1;
    }
    if (!expect(loaded_buffer->metadata("origin.file.format") == "torch-tensor",
                "TorchFileSrc did not stamp file.format metadata")) {
        return 1;
    }
    if (!expect(loaded_buffer->metadata("origin.file.path") == saved_path.string(),
                "TorchFileSrc did not stamp file.path metadata")) {
        return 1;
    }
    if (!expect(!loaded_buffer->has_metadata("payload.leakage.inverted"),
                "TorchFileSrc should not stamp leakage.inverted metadata; the user sets it")) {
        return 1;
    }
    if (!expect(!loaded_buffer->has_metadata("capture.sample_rate_hz"),
                "TorchFileSrc still stamped default sample_rate_hz metadata")) {
        return 1;
    }

    const auto loaded_payload = loaded_buffer->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(loaded_payload != nullptr, "TorchFileSrc payload type was wrong")) {
        return 1;
    }
    if (!expect(loaded_payload->dtype() == saved_payload->dtype(), "loaded tensor dtype changed")) {
        return 1;
    }
    if (!expect(loaded_payload->shape()[0] == saved_payload->shape()[0] &&
                    loaded_payload->shape()[1] == saved_payload->shape()[1],
                "loaded tensor shape changed")) {
        return 1;
    }
    if (!expect(torch::equal(loaded_payload->tensor(), saved_payload->tensor()), "loaded tensor values changed")) {
        return 1;
    }

    base_plugin::TorchFileSrc inverted_source;
    inverted_source.set_property("path", saved_path.string());
    inverted_source.set_property("invert", true);
    auto inverted_buffer = inverted_source.process(std::nullopt);
    if (!expect(inverted_buffer.has_value(), "TorchFileSrc inverted load did not produce a buffer")) {
        return 1;
    }
    if (!expect(!inverted_buffer->has_metadata("payload.leakage.inverted"),
                "TorchFileSrc invert should not stamp leakage.inverted metadata")) {
        return 1;
    }
    const auto inverted_payload = inverted_buffer->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(inverted_payload != nullptr, "TorchFileSrc inverted payload type was wrong")) {
        return 1;
    }
    if (!expect(torch::equal(inverted_payload->tensor(), -saved_payload->tensor()),
                "TorchFileSrc invert did not flip leakage tensor values")) {
        return 1;
    }

    base_plugin::TorchFileSink empty_path_sink;
    if (!expect(throws_exception<std::invalid_argument>(
                    [&empty_path_sink, &torch_buffer] { (void)empty_path_sink.process(torch_buffer); }),
                "TorchFileSink accepted an empty path")) {
        return 1;
    }

    base_plugin::TorchFileSrc empty_path_source;
    if (!expect(throws_exception<std::invalid_argument>(
                    [&empty_path_source] { (void)empty_path_source.process(std::nullopt); }),
                "TorchFileSrc accepted an empty path")) {
        return 1;
    }

    const auto descriptors = base_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "base plugin descriptor count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].name == "leakflow_plugins_base", "base plugin descriptor name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].version == "0.10", "base plugin descriptor version was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].purpose == "Leakflow base elements", "base plugin descriptor purpose was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 5, "base element descriptor count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].type_name == "TorchFileSrc", "TorchFileSrc descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].long_name == "Torch File Source",
                "TorchFileSrc descriptor long name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].rank == leakflow::ElementRank::Primary,
                "TorchFileSrc descriptor rank was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].klass == "Source/File/Torch",
                "TorchFileSrc descriptor klass was wrong")) {
        return 1;
    }
    const auto* torch_file_src_device_property = find_property(descriptors[0].elements[0].property_specs, "device");
    if (!expect(torch_file_src_device_property != nullptr, "TorchFileSrc device property was missing")) {
        return 1;
    }
    if (!expect(torch_file_src_device_property->value_hint.find("cuda:0") != std::string::npos,
                "TorchFileSrc device value hint does not mention CUDA devices")) {
        return 1;
    }
    const auto& torch_file_src_descriptor = descriptors[0].elements[0];
    const auto* file_path_metadata =
        find_metadata(torch_file_src_descriptor.metadata_set_by_element, "origin.file.path");
    if (!expect(file_path_metadata != nullptr, "TorchFileSrc file.path metadata descriptor was missing")) {
        return 1;
    }
    if (!expect(leakflow::property_value_type_name(file_path_metadata->value_type) == "string",
            "TorchFileSrc file.path metadata type was wrong")) {
        return 1;
    }
    if (!expect(file_path_metadata->example_values.size() == 1
                && file_path_metadata->example_values[0] == "traces/aes/sync/aes_sync_poi/key_01/traces.pt",
            "TorchFileSrc file.path metadata example was wrong")) {
        return 1;
    }
    const auto* file_size_metadata =
        find_metadata(torch_file_src_descriptor.metadata_set_by_element, "origin.file.size");
    if (!expect(file_size_metadata != nullptr, "TorchFileSrc file.size metadata descriptor was missing")) {
        return 1;
    }
    if (!expect(leakflow::property_value_type_name(file_size_metadata->value_type) == "integer",
            "TorchFileSrc file.size metadata type was wrong")) {
        return 1;
    }
    if (!expect(file_size_metadata->example_values.size() == 1
                && file_size_metadata->example_values[0] == "40001506",
            "TorchFileSrc file.size metadata example was wrong")) {
        return 1;
    }
    const auto* stamped_sample_rate_metadata =
        find_metadata(torch_file_src_descriptor.metadata_set_by_element, "capture.sample_rate_hz");
    if (!expect(stamped_sample_rate_metadata == nullptr,
                "TorchFileSrc still advertised sample_rate_hz as stamped metadata")) {
        return 1;
    }
    const auto* sample_rate_metadata =
        find_metadata(torch_file_src_descriptor.metadata_suggestions, "capture.sample_rate_hz");
    if (!expect(sample_rate_metadata != nullptr, "TorchFileSrc sample_rate_hz suggestion was missing")) {
        return 1;
    }
    if (!expect(sample_rate_metadata->details.find("TracePlot") != std::string::npos,
                "TorchFileSrc sample_rate_hz suggestion details did not mention TracePlot")) {
        return 1;
    }
    const auto* capture_source_metadata =
        find_metadata(torch_file_src_descriptor.metadata_suggestions, "capture.source");
    if (!expect(capture_source_metadata != nullptr, "TorchFileSrc capture.source suggestion was missing")) {
        return 1;
    }
    if (!expect(capture_source_metadata->example_values.size() == 1
                && capture_source_metadata->example_values[0] == "ChipWhisperer",
            "TorchFileSrc capture.source example was wrong")) {
        return 1;
    }
    const auto* leakage_range_metadata =
        find_metadata(torch_file_src_descriptor.metadata_suggestions, "payload.leakage.range");
    if (!expect(leakage_range_metadata != nullptr, "TorchFileSrc leakage.range suggestion was missing")) {
        return 1;
    }
    if (!expect(leakflow::property_value_type_name(leakage_range_metadata->value_type) == "double interval",
            "TorchFileSrc leakage.range metadata type was wrong")) {
        return 1;
    }
    if (!expect(leakage_range_metadata->example_values.size() == 1
                && leakage_range_metadata->example_values[0] == "[-0.5,0.5]",
            "TorchFileSrc leakage.range example was wrong")) {
        return 1;
    }
    base_plugin::TorchFileSrc cuda_device_source;
    try {
        cuda_device_source.set_property("device", std::string("cuda:0"));
    } catch (const std::exception&) {
        std::cerr << "TorchFileSrc rejected a CUDA device string\n";
        return 1;
    }
    if (!expect(descriptors[0].elements[1].type_name == "TorchConvert",
                "TorchConvert descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[2].type_name == "TorchFileSink",
                "TorchFileSink descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(base_plugin::find_plugin_descriptor("leakflow_plugins_base") != nullptr,
                "base plugin descriptor was not findable")) {
        return 1;
    }

    std::filesystem::remove(saved_path);
    return 0;
}
