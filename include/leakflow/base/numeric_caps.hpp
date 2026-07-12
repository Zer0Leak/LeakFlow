#pragma once

#include "leakflow/core/caps.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace leakflow::base {

inline constexpr auto torch_tensor_caps_type = "leakflow/torch-tensor";

inline constexpr auto caps_param_dtype = "dtype";
inline constexpr auto caps_param_device = "device";
inline constexpr auto caps_param_rank = "rank";
inline constexpr auto caps_param_shape = "shape";

inline constexpr auto cpu_device_caps_value = "cpu";

[[nodiscard]] std::string shape_caps_value(const std::int64_t* values, std::size_t count);
[[nodiscard]] std::string shape_caps_value(const std::uint64_t* values, std::size_t count);
[[nodiscard]] std::string generic_rank_layout(std::uint64_t rank);

[[nodiscard]] Caps numeric_array_caps(
    std::string type,
    std::string dtype,
    std::string device,
    std::uint64_t rank,
    std::string shape);

} // namespace leakflow::base
