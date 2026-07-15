#include "leakflow/core/property.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace leakflow {
namespace {

template <typename T>
[[nodiscard]] bool holds(const PropertyValue& value)
{
    return std::holds_alternative<T>(value);
}

template <typename T>
void append_list(std::ostringstream& output, const std::vector<T>& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

void append_list(std::ostringstream& output, const StringList& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

[[nodiscard]] bool string_allowed(std::string_view value, const StringEnumConstraint& constraint)
{
    return std::ranges::any_of(constraint.allowed_values, [value](const std::string& allowed) {
        return allowed == value;
    });
}

void validate_constraint_shape(const PropertySpec& spec)
{
    std::visit(
        [&spec](const auto& constraint) {
            using Constraint = std::decay_t<decltype(constraint)>;

            if constexpr (std::is_same_v<Constraint, std::monostate>) {
                return;
            } else if constexpr (std::is_same_v<Constraint, IntRangeConstraint>) {
                if (constraint.min > constraint.max) {
                    throw std::invalid_argument("integer range constraint min cannot exceed max");
                }
                if (!holds<std::int64_t>(spec.default_value)) {
                    throw std::invalid_argument("integer range constraint requires integer property default");
                }
            } else if constexpr (std::is_same_v<Constraint, DoubleRangeConstraint>) {
                if (constraint.min > constraint.max) {
                    throw std::invalid_argument("double range constraint min cannot exceed max");
                }
                if (!holds<double>(spec.default_value)) {
                    throw std::invalid_argument("double range constraint requires double property default");
                }
            } else if constexpr (std::is_same_v<Constraint, StringEnumConstraint>) {
                if (constraint.allowed_values.empty()) {
                    throw std::invalid_argument("string enum constraint requires at least one allowed value");
                }
                if (!holds<std::string>(spec.default_value)) {
                    throw std::invalid_argument("string enum constraint requires string property default");
                }
            }
        },
        spec.constraint);
}

void validate_constraint_value(const PropertySpec& spec, const PropertyValue& value)
{
    std::visit(
        [&value](const auto& constraint) {
            using Constraint = std::decay_t<decltype(constraint)>;

            if constexpr (std::is_same_v<Constraint, std::monostate>) {
                return;
            } else if constexpr (std::is_same_v<Constraint, IntRangeConstraint>) {
                const auto& integer = std::get<std::int64_t>(value);
                if (integer < constraint.min || integer > constraint.max) {
                    throw std::invalid_argument("integer property value violates range constraint");
                }
            } else if constexpr (std::is_same_v<Constraint, DoubleRangeConstraint>) {
                const auto& number = std::get<double>(value);
                if (number < constraint.min || number > constraint.max) {
                    throw std::invalid_argument("double property value violates range constraint");
                }
            } else if constexpr (std::is_same_v<Constraint, StringEnumConstraint>) {
                const auto& text = std::get<std::string>(value);
                if (!string_allowed(text, constraint)) {
                    throw std::invalid_argument("string property value violates enum constraint");
                }
            }
        },
        spec.constraint);
}

[[nodiscard]] std::string lower_copy(std::string_view text)
{
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return out;
}

[[nodiscard]] std::string trim_copy(std::string_view text)
{
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(begin, end - begin + 1));
}

[[nodiscard]] std::optional<Color> named_color_value(std::string_view lowered)
{
    static constexpr std::array<std::pair<std::string_view, std::array<float, 3>>, 15> table{{
        {"black", {0.0F, 0.0F, 0.0F}},
        {"blue", {0.0F, 0.0F, 1.0F}},
        {"cyan", {0.0F, 1.0F, 1.0F}},
        {"gray", {0.5F, 0.5F, 0.5F}},
        {"grey", {0.5F, 0.5F, 0.5F}},
        {"green", {0.0F, 0.5F, 0.0F}},
        {"lime", {0.0F, 1.0F, 0.0F}},
        {"magenta", {1.0F, 0.0F, 1.0F}},
        {"orange", {1.0F, 0.64705884F, 0.0F}},
        {"pink", {1.0F, 0.75294119F, 0.79607844F}},
        {"purple", {0.50196081F, 0.0F, 0.50196081F}},
        {"red", {1.0F, 0.0F, 0.0F}},
        {"teal", {0.0F, 0.50196081F, 0.50196081F}},
        {"white", {1.0F, 1.0F, 1.0F}},
        {"yellow", {1.0F, 1.0F, 0.0F}},
    }};

    for (const auto& [name, rgb] : table) {
        if (lowered == name) {
            return Color{rgb[0], rgb[1], rgb[2], 1.0F};
        }
    }
    return std::nullopt;
}

[[nodiscard]] int hex_digit_value(char character)
{
    if (character >= '0' && character <= '9') {
        return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
        return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F') {
        return character - 'A' + 10;
    }
    return -1;
}

[[nodiscard]] std::optional<float> hex_pair(std::string_view text, std::size_t index)
{
    const auto high = hex_digit_value(text[index]);
    const auto low = hex_digit_value(text[index + 1]);
    if (high < 0 || low < 0) {
        return std::nullopt;
    }
    return static_cast<float>((high << 4) + low) / 255.0F;
}

[[nodiscard]] std::optional<Color> parse_hex_color(std::string_view text)
{
    std::string_view body = text;
    if (!body.empty() && body.front() == '#') {
        body.remove_prefix(1);
    }
    if (body.size() != 6 && body.size() != 8) {
        return std::nullopt;
    }

    const auto r = hex_pair(body, 0);
    const auto g = hex_pair(body, 2);
    const auto b = hex_pair(body, 4);
    if (!r || !g || !b) {
        return std::nullopt;
    }
    Color color{*r, *g, *b, 1.0F};
    if (body.size() == 8) {
        const auto alpha = hex_pair(body, 6);
        if (!alpha) {
            return std::nullopt;
        }
        color.a = *alpha;
    }
    return color;
}

[[nodiscard]] std::optional<double> parse_number(const std::string& text)
{
    if (text.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size()) {
        return std::nullopt;
    }
    return value;
}

[[nodiscard]] std::optional<Color> parse_function_color(std::string_view text)
{
    const auto open = text.find('(');
    const auto close = text.rfind(')');
    if (open == std::string_view::npos || close == std::string_view::npos || close < open) {
        return std::nullopt;
    }

    const auto inner = text.substr(open + 1, close - open - 1);
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= inner.size()) {
        const auto comma = inner.find(',', begin);
        const auto piece = comma == std::string_view::npos ? inner.substr(begin) : inner.substr(begin, comma - begin);
        parts.push_back(trim_copy(piece));
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
    }

    if (parts.size() != 3 && parts.size() != 4) {
        return std::nullopt;
    }

    Color color;
    const std::array<float*, 3> channels{&color.r, &color.g, &color.b};
    for (std::size_t index = 0; index < 3; ++index) {
        const auto number = parse_number(parts[index]);
        if (!number) {
            return std::nullopt;
        }
        *channels[index] = static_cast<float>(std::clamp(*number / 255.0, 0.0, 1.0));
    }

    color.a = 1.0F;
    if (parts.size() == 4) {
        const auto alpha = parse_number(parts[3]);
        if (!alpha) {
            return std::nullopt;
        }
        color.a = static_cast<float>(std::clamp(*alpha, 0.0, 1.0));
    }
    return color;
}

} // namespace

std::optional<Color> parse_color(std::string_view text)
{
    const auto trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    const auto lowered = lower_copy(trimmed);
    if (const auto named = named_color_value(lowered)) {
        return named;
    }
    if (lowered.rfind("rgb(", 0) == 0 || lowered.rfind("rgba(", 0) == 0) {
        return parse_function_color(trimmed);
    }
    return parse_hex_color(trimmed);
}

std::string color_to_string(const Color& color)
{
    const auto to_byte = [](float component) {
        return static_cast<int>(std::lround(std::clamp(component, 0.0F, 1.0F) * 255.0F));
    };

    std::ostringstream output;
    output << '#' << std::uppercase << std::hex << std::setfill('0');
    output << std::setw(2) << to_byte(color.r) << std::setw(2) << to_byte(color.g) << std::setw(2) << to_byte(color.b);
    if (std::abs(color.a - 1.0F) > 1.0e-6F) {
        output << std::setw(2) << to_byte(color.a);
    }
    return output.str();
}

PropertySpec::PropertySpec(std::string name,
    PropertyValue default_value,
    std::string description,
    std::string unit,
    PropertyConstraint constraint,
    std::string value_hint,
    PropertyEffect effect,
    bool optional,
    bool writable)
    : name(std::move(name))
    , default_value(std::move(default_value))
    , description(std::move(description))
    , unit(std::move(unit))
    , constraint(std::move(constraint))
    , value_hint(std::move(value_hint))
    , effect(std::move(effect))
    , optional(optional)
    , writable(writable)
{
    validate_property_spec(*this);
}

bool property_value_is_null(const PropertyValue& value)
{
    return std::holds_alternative<std::monostate>(value);
}

std::string property_value_type_name(const PropertyValue& value)
{
    return std::visit(
        [](const auto& typed_value) -> std::string {
            using Value = std::decay_t<decltype(typed_value)>;

            if constexpr (std::is_same_v<Value, bool>) {
                return "bool";
            } else if constexpr (std::is_same_v<Value, std::int64_t>) {
                return "integer";
            } else if constexpr (std::is_same_v<Value, double>) {
                return "double";
            } else if constexpr (std::is_same_v<Value, std::string>) {
                return "string";
            } else if constexpr (std::is_same_v<Value, IntInterval>) {
                return "integer interval";
            } else if constexpr (std::is_same_v<Value, DoubleInterval>) {
                return "double interval";
            } else if constexpr (std::is_same_v<Value, IntList>) {
                return "integer list";
            } else if constexpr (std::is_same_v<Value, DoubleList>) {
                return "double list";
            } else if constexpr (std::is_same_v<Value, StringList>) {
                return "string list";
            } else if constexpr (std::is_same_v<Value, Units>) {
                return "units";
            } else if constexpr (std::is_same_v<Value, Color>) {
                return "color";
            } else if constexpr (std::is_same_v<Value, std::monostate>) {
                return "null";
            }
        },
        value);
}

std::string property_value_to_string(const PropertyValue& value)
{
    return std::visit(
        [](const auto& typed_value) -> std::string {
            using Value = std::decay_t<decltype(typed_value)>;
            std::ostringstream output;

            if constexpr (std::is_same_v<Value, bool>) {
                return typed_value ? "true" : "false";
            } else if constexpr (std::is_same_v<Value, std::int64_t>) {
                output << typed_value;
            } else if constexpr (std::is_same_v<Value, double>) {
                output << typed_value;
            } else if constexpr (std::is_same_v<Value, std::string>) {
                output << typed_value;
            } else if constexpr (std::is_same_v<Value, IntInterval>) {
                output << typed_value.begin << ".." << typed_value.end;
            } else if constexpr (std::is_same_v<Value, DoubleInterval>) {
                output << typed_value.begin << ".." << typed_value.end;
            } else if constexpr (std::is_same_v<Value, IntList>
                || std::is_same_v<Value, DoubleList>
                || std::is_same_v<Value, StringList>) {
                append_list(output, typed_value);
            } else if constexpr (std::is_same_v<Value, Units>) {
                return typed_value.format();
            } else if constexpr (std::is_same_v<Value, Color>) {
                return color_to_string(typed_value);
            } else if constexpr (std::is_same_v<Value, std::monostate>) {
                return "null";
            }

            return output.str();
        },
        value);
}

std::string_view property_effect_kind_name(PropertyEffectKind kind)
{
    switch (kind) {
    case PropertyEffectKind::UiControl:
        return "ui-control";
    case PropertyEffectKind::SinkDisplay:
        return "sink-display";
    case PropertyEffectKind::MetadataOutput:
        return "metadata-output";
    case PropertyEffectKind::PayloadOutput:
        return "payload-output";
    case PropertyEffectKind::CapsOutput:
        return "caps-output";
    case PropertyEffectKind::Lifecycle:
        return "lifecycle";
    }

    return "unknown";
}

std::string_view property_invalidation_scope_name(PropertyInvalidationScope scope)
{
    switch (scope) {
    case PropertyInvalidationScope::None:
        return "none";
    case PropertyInvalidationScope::ElementUi:
        return "element-ui";
    case PropertyInvalidationScope::Downstream:
        return "downstream";
    case PropertyInvalidationScope::FullPipeline:
        return "full-pipeline";
    }

    return "unknown";
}

void validate_property_spec(const PropertySpec& spec)
{
    if (spec.name.empty()) {
        throw std::invalid_argument("property name cannot be empty");
    }

    validate_constraint_shape(spec);
    validate_property_effect(spec.effect);
    validate_property_value(spec, spec.default_value);
}

void validate_property_value(const PropertySpec& spec, const PropertyValue& value)
{
    if (property_value_is_null(value)) {
        if (!spec.optional) {
            throw std::invalid_argument("null value is only allowed for optional properties");
        }
        return;
    }

    if (value.index() != spec.default_value.index()) {
        throw std::invalid_argument("property value type does not match property default type");
    }

    validate_constraint_value(spec, value);
}

void validate_property_effect(const PropertyEffect& effect)
{
    for (const auto& pad_name : effect.output_pads) {
        if (pad_name.empty()) {
            throw std::invalid_argument("property effect output pad name cannot be empty");
        }
    }

    if (effect.kind == PropertyEffectKind::UiControl && effect.scope != PropertyInvalidationScope::None
        && effect.scope != PropertyInvalidationScope::ElementUi) {
        throw std::invalid_argument("ui-control property effects cannot invalidate pipeline dataflow");
    }
    if (effect.kind == PropertyEffectKind::SinkDisplay && effect.scope != PropertyInvalidationScope::ElementUi
        && effect.scope != PropertyInvalidationScope::None) {
        throw std::invalid_argument("sink-display property effects cannot invalidate upstream dataflow");
    }
    if (effect.kind == PropertyEffectKind::Lifecycle && effect.scope != PropertyInvalidationScope::FullPipeline) {
        throw std::invalid_argument("lifecycle property effects require full-pipeline invalidation");
    }
    if ((effect.kind == PropertyEffectKind::MetadataOutput || effect.kind == PropertyEffectKind::PayloadOutput
            || effect.kind == PropertyEffectKind::CapsOutput)
        && effect.scope == PropertyInvalidationScope::FullPipeline) {
        throw std::invalid_argument("output property effects should invalidate downstream outputs, not the full pipeline");
    }
    if ((effect.kind == PropertyEffectKind::MetadataOutput || effect.kind == PropertyEffectKind::PayloadOutput
            || effect.kind == PropertyEffectKind::CapsOutput)
        && effect.scope != PropertyInvalidationScope::Downstream) {
        throw std::invalid_argument("output property effects require downstream invalidation");
    }
}

} // namespace leakflow
