#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace leakflow {

struct IntInterval {
    std::int64_t begin;
    std::int64_t end;

    [[nodiscard]] bool operator==(const IntInterval&) const = default;
};

struct DoubleInterval {
    double begin;
    double end;

    [[nodiscard]] bool operator==(const DoubleInterval&) const = default;
};

using IntList = std::vector<std::int64_t>;
using DoubleList = std::vector<double>;
using StringList = std::vector<std::string>;

// RGBA color in normalized [0,1] components. "Unset"/auto is expressed by making
// the property optional (a null PropertyValue), not by a sentinel color.
struct Color {
    float r = 0.0F;
    float g = 0.0F;
    float b = 0.0F;
    float a = 0.0F;

    [[nodiscard]] bool operator==(const Color&) const = default;
};

// std::monostate is the "null" / unset value of an optional property (see
// PropertySpec::optional).
using PropertyValue = std::variant<
    bool,
    std::int64_t,
    double,
    std::string,
    IntInterval,
    DoubleInterval,
    IntList,
    DoubleList,
    StringList,
    Color,
    std::monostate>;

[[nodiscard]] bool property_value_is_null(const PropertyValue& value);

// Color value helpers. parse_color accepts named colors (red, blue, ...),
// #RRGGBB / #RRGGBBAA hex, and rgb(...)/rgba(...); it returns nullopt for
// unparseable text. color_to_string produces #RRGGBB or #RRGGBBAA.
[[nodiscard]] std::optional<Color> parse_color(std::string_view text);
[[nodiscard]] std::string color_to_string(const Color& color);

struct IntRangeConstraint {
    std::int64_t min;
    std::int64_t max;

    [[nodiscard]] bool operator==(const IntRangeConstraint&) const = default;
};

struct DoubleRangeConstraint {
    double min;
    double max;

    [[nodiscard]] bool operator==(const DoubleRangeConstraint&) const = default;
};

struct StringEnumConstraint {
    std::vector<std::string> allowed_values;

    [[nodiscard]] bool operator==(const StringEnumConstraint&) const = default;
};

using PropertyConstraint = std::variant<
    std::monostate,
    IntRangeConstraint,
    DoubleRangeConstraint,
    StringEnumConstraint>;

enum class PropertyEffectKind {
    UiOnly,
    SinkDisplay,
    MetadataOutput,
    PayloadOutput,
    CapsOutput,
    Lifecycle,
};

enum class PropertyInvalidationScope {
    None,
    ElementUi,
    Downstream,
    FullPipeline,
};

struct PropertyEffect {
    PropertyEffectKind kind = PropertyEffectKind::UiOnly;
    PropertyInvalidationScope scope = PropertyInvalidationScope::None;
    std::vector<std::string> output_pads;

    [[nodiscard]] bool operator==(const PropertyEffect&) const = default;
};

struct PropertySpec {
    PropertySpec(std::string name,
        PropertyValue default_value,
        std::string description = {},
        std::string unit = {},
        PropertyConstraint constraint = std::monostate{},
        std::string value_hint = {},
        PropertyEffect effect = {},
        bool optional = false);

    std::string name;
    PropertyValue default_value;
    std::string description;
    std::string unit;
    PropertyConstraint constraint;
    std::string value_hint;
    PropertyEffect effect;
    // When true, the property may be null (std::monostate). default_value is the
    // type exemplar (and the value used when filling in a null); an optional
    // property starts null.
    bool optional = false;
};

[[nodiscard]] std::string property_value_type_name(const PropertyValue& value);
[[nodiscard]] std::string property_value_to_string(const PropertyValue& value);
[[nodiscard]] std::string_view property_effect_kind_name(PropertyEffectKind kind);
[[nodiscard]] std::string_view property_invalidation_scope_name(PropertyInvalidationScope scope);

void validate_property_spec(const PropertySpec& spec);
void validate_property_value(const PropertySpec& spec, const PropertyValue& value);
void validate_property_effect(const PropertyEffect& effect);

template <typename T>
[[nodiscard]] std::optional<T> property_value_as(const PropertyValue& value)
{
    if (const auto* typed_value = std::get_if<T>(&value)) {
        return *typed_value;
    }

    return std::nullopt;
}

} // namespace leakflow
