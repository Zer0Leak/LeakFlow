#pragma once

#include "leakflow/core/buffer.hpp"

#include <cstdint>
#include <string>

namespace leakflow::plugins::core {

[[nodiscard]] std::string summarize_buffer(const Buffer& buffer, std::int64_t summary_level = 1);

} // namespace leakflow::plugins::core

