#pragma once

#include "leakflow/core/element.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace leakflow::plugins::core {

[[nodiscard]] inline std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] inline std::int64_t int_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }

    return fallback;
}

[[nodiscard]] inline bool bool_property_or(const Element& element, std::string_view name, bool fallback)
{
    if (const auto value = element.property_as<bool>(name)) {
        return *value;
    }

    return fallback;
}

} // namespace leakflow::plugins::core

