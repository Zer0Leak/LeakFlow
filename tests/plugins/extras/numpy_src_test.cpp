#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/extras/numpy_payload.hpp"
#include "leakflow/plugins/extras/descriptor_catalog.hpp"
#include "leakflow/plugins/extras/extras_elements.hpp"

#include <cnpy++.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
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

std::filesystem::path temp_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

} // namespace

int main()
{
    namespace extras_plugin = leakflow::plugins::extras;

    const auto traces_path = temp_path("leakflow_numpy_src_float.npy");
    const std::vector<float> traces{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    cnpypp::npy_save(traces_path.string(), traces.begin(), {2, 3});

    extras_plugin::NumpySrc source;
    source.set_property("path", traces_path.string());

    auto output = source.process(std::nullopt);
    if (!expect(output.has_value(), "NumpySrc did not produce a buffer")) {
        return 1;
    }
    if (!expect(output->caps().type() == "leakflow/numpy-array", "NumpySrc emitted wrong buffer caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_dtype) == "float32",
                "NumpySrc emitted wrong buffer dtype caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_device) == "cpu",
                "NumpySrc emitted wrong buffer device caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_rank) == "2",
                "NumpySrc emitted wrong buffer rank caps")) {
        return 1;
    }
    if (!expect(output->caps().param(leakflow::base::caps_param_shape) == "[2,3]",
                "NumpySrc emitted wrong buffer shape caps")) {
        return 1;
    }
    if (!expect(source.output_pads().size() == 1, "NumpySrc output pad count changed")) {
        return 1;
    }
    if (!expect(source.output_pads()[0].caps().type() == "leakflow/numpy-array",
                "NumpySrc output pad caps were wrong")) {
        return 1;
    }
    if (!expect(output->metadata("routing.element") == "numpysrc0", "NumpySrc did not stamp element metadata")) {
        return 1;
    }
    if (!expect(output->metadata("origin.file.path") == traces_path.string(), "NumpySrc did not stamp file.path metadata")) {
        return 1;
    }
    if (!expect(output->metadata("origin.file.size") == std::to_string(std::filesystem::file_size(traces_path)),
                "NumpySrc did not stamp file.size metadata")) {
        return 1;
    }
    if (!expect(output->metadata("payload.layout") == "axis_0/axis_1",
                "NumpySrc did not publish the generic array layout")) {
        return 1;
    }

    const auto payload = output->payload_as<leakflow::extras::NumpyPayload>();
    if (!expect(payload != nullptr, "NumpySrc payload type was wrong")) {
        return 1;
    }
    if (!expect(payload->type_name() == "leakflow/numpy-array", "NumpyPayload type name changed")) {
        return 1;
    }
    if (!expect(payload->shape() == std::vector<std::uint64_t>{2, 3}, "NumpySrc generated fixture shape mismatch")) {
        return 1;
    }
    if (!expect(payload->word_size() == sizeof(float), "NumpySrc generated fixture word size mismatch")) {
        return 1;
    }
    if (!expect(payload->dtype_name() == "float32", "NumpySrc generated fixture dtype mismatch")) {
        return 1;
    }

    extras_plugin::NumpySrc empty_path_source;
    if (!expect(throws_exception<std::invalid_argument>(
                    [&empty_path_source] { (void)empty_path_source.process(std::nullopt); }),
                "NumpySrc accepted an empty path")) {
        return 1;
    }

    const auto descriptors = extras_plugin::plugin_descriptors();
    if (!expect(descriptors.size() == 1, "extras plugin descriptor count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].name == "leakflow_plugins_extras", "extras plugin descriptor name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements.size() == 6, "extras element descriptor count was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].type_name == "NumpySrc", "NumpySrc descriptor type name was wrong")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].output_pads.size() == 1, "NumpySrc descriptor output pad count changed")) {
        return 1;
    }
    if (!expect(descriptors[0].elements[0].output_pads[0].caps().type() == "leakflow/numpy-array",
                "NumpySrc descriptor output pad caps were wrong")) {
        return 1;
    }
    if (!expect(extras_plugin::find_plugin_descriptor("leakflow_plugins_extras") != nullptr,
                "extras plugin descriptor was not findable")) {
        return 1;
    }
    if (!expect(extras_plugin::find_plugin_descriptor("missing") == nullptr,
                "missing extras plugin descriptor was unexpectedly found")) {
        return 1;
    }

    std::filesystem::remove(traces_path);
    return 0;
}
