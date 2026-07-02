#include "leakflow/base/numeric_caps.hpp"

#include <sstream>
#include <utility>

namespace leakflow::base {
namespace {

template <typename T>
[[nodiscard]] std::string shape_caps_value_from(const T* values, std::size_t count)
{
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
    return output.str();
}

} // namespace

std::string shape_caps_value(const std::int64_t* values, std::size_t count)
{
    return shape_caps_value_from(values, count);
}

std::string shape_caps_value(const std::uint64_t* values, std::size_t count)
{
    return shape_caps_value_from(values, count);
}

Caps numeric_array_caps(
    std::string type,
    std::string dtype,
    std::string device,
    std::uint64_t rank,
    std::string shape)
{
    Caps caps(std::move(type));

    if (!dtype.empty()) {
        caps.set_param(caps_param_dtype, std::move(dtype));
    }
    if (!device.empty()) {
        caps.set_param(caps_param_device, std::move(device));
    }
    caps.set_param(caps_param_rank, std::to_string(rank));
    caps.set_param(caps_param_shape, std::move(shape));

    return caps;
}

} // namespace leakflow::base
