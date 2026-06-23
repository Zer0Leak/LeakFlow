#pragma once

#include <fmt/color.h>

#include <string>
#include <string_view>

namespace leakflow::render {

enum class ColorMode {
    Auto,
    Never,
    Always,
};

enum class GlyphMode {
    Ascii,
    Utf8,
};

enum class StyleRole {
    Heading,
    Section,
    Label,
    Value,
    Type,
    Tree,
    Feature,
    Path,
    Number,
    Size,
    Boolean,
    Warning,
};

struct TerminalTheme {
    fmt::text_style heading;
    fmt::text_style section;
    fmt::text_style label;
    fmt::text_style value;
    fmt::text_style type;
    fmt::text_style tree;
    fmt::text_style feature;
    fmt::text_style path;
    fmt::text_style number;
    fmt::text_style size;
    fmt::text_style boolean;
    fmt::text_style warning;

    [[nodiscard]] static TerminalTheme leakflow();
};

struct GlyphSet {
    std::string branch;
    std::string leaf;
    std::string vertical;
    std::string space;
    std::string bullet;
    std::string arrow;
    std::string ok;
    std::string none;

    [[nodiscard]] static GlyphSet ascii();
    [[nodiscard]] static GlyphSet utf8();
};

class TerminalStyler {
public:
    explicit TerminalStyler(ColorMode color_mode = ColorMode::Auto);
    TerminalStyler(ColorMode color_mode, TerminalTheme theme);

    [[nodiscard]] bool enabled() const;
    [[nodiscard]] std::string apply(StyleRole role, std::string_view text) const;

private:
    [[nodiscard]] const fmt::text_style& style_for(StyleRole role) const;

    bool enabled_ = false;
    TerminalTheme theme_;
};

[[nodiscard]] bool stdout_is_terminal();
[[nodiscard]] ColorMode parse_color_mode(std::string_view text);
[[nodiscard]] GlyphMode parse_glyph_mode(std::string_view text);
[[nodiscard]] GlyphSet glyphs_for(GlyphMode mode);

} // namespace leakflow::render
