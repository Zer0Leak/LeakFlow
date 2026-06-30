#pragma once

#include <map>
#include <string>
#include <string_view>

namespace leakflow {

inline constexpr auto generic_buffer_caps_type = "leakflow/buffer";
inline constexpr auto any_caps_type = "ANY";

class Caps {
public:
    using Params = std::map<std::string, std::string>;

    explicit Caps(std::string type);
    Caps(std::string type, Params params);

    [[nodiscard]] const std::string& type() const;

    [[nodiscard]] bool has_param(std::string_view key) const;
    [[nodiscard]] const std::string& param(std::string_view key) const;
    [[nodiscard]] std::string param_or(std::string_view key, std::string default_value) const;

    void set_param(std::string key, std::string value);

    [[nodiscard]] const Params& params() const;

    [[nodiscard]] std::string to_string() const;

private:
    std::string type_;
    Params params_;
};

[[nodiscard]] bool caps_are_compatible(const Caps& source, const Caps& sink);
[[nodiscard]] bool caps_are_any(const Caps& caps);

} // namespace leakflow
