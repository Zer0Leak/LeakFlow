#include "leakflow/render/terminal_style.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace leakflow::render {
namespace {

[[nodiscard]] bool terminal_is_tty()
{
#ifdef _WIN32
    return _isatty(_fileno(stdout)) == 1;
#else
    return isatty(STDOUT_FILENO) == 1;
#endif
}

[[nodiscard]] bool color_enabled(ColorMode mode)
{
    switch (mode) {
    case ColorMode::Never:
        return false;
    case ColorMode::Always:
        return true;
    case ColorMode::Auto:
        return terminal_is_tty();
    }

    return false;
}

[[nodiscard]] std::string normalized(std::string_view text)
{
    std::string result;
    result.reserve(text.size());
    for (const auto character : text) {
        if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
            result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
        }
    }
    return result;
}

} // namespace

TerminalTheme TerminalTheme::leakflow()
{
    return {
        .heading = fmt::fg(fmt::color::yellow) | fmt::emphasis::bold,
        .section = fmt::fg(fmt::color::yellow),
        .label = fmt::fg(fmt::terminal_color::bright_blue),
        .value = fmt::fg(fmt::color::cyan),
        .type = fmt::fg(fmt::color::green),
        .tree = fmt::fg(fmt::color::magenta),
        .feature = fmt::fg(fmt::color::green),
        .path = fmt::fg(fmt::color::cyan),
        .number = fmt::fg(fmt::color::cyan),
        .size = fmt::fg(fmt::color::cyan),
        .boolean = fmt::fg(fmt::color::green),
        .warning = fmt::fg(fmt::color::yellow) | fmt::emphasis::bold,
    };
}

GlyphSet GlyphSet::ascii()
{
    return {
        .branch = "+-",
        .leaf = "`-",
        .vertical = "|",
        .space = " ",
        .bullet = "-",
        .arrow = "->",
        .ok = "ok",
        .none = "none",
    };
}

GlyphSet GlyphSet::utf8()
{
    return {
        .branch = "├─",
        .leaf = "└─",
        .vertical = "│",
        .space = " ",
        .bullet = "•",
        .arrow = "→",
        .ok = "✓",
        .none = "∅",
    };
}

TerminalStyler::TerminalStyler(ColorMode color_mode)
    : TerminalStyler(color_mode, TerminalTheme::leakflow())
{
}

TerminalStyler::TerminalStyler(ColorMode color_mode, TerminalTheme theme)
    : enabled_(color_enabled(color_mode))
    , theme_(std::move(theme))
{
}

bool TerminalStyler::enabled() const
{
    return enabled_;
}

std::string TerminalStyler::apply(StyleRole role, std::string_view text) const
{
    if (!enabled_) {
        return std::string(text);
    }

    return fmt::format(style_for(role), "{}", text);
}

const fmt::text_style& TerminalStyler::style_for(StyleRole role) const
{
    switch (role) {
    case StyleRole::Heading:
        return theme_.heading;
    case StyleRole::Section:
        return theme_.section;
    case StyleRole::Label:
        return theme_.label;
    case StyleRole::Value:
        return theme_.value;
    case StyleRole::Type:
        return theme_.type;
    case StyleRole::Tree:
        return theme_.tree;
    case StyleRole::Feature:
        return theme_.feature;
    case StyleRole::Path:
        return theme_.path;
    case StyleRole::Number:
        return theme_.number;
    case StyleRole::Size:
        return theme_.size;
    case StyleRole::Boolean:
        return theme_.boolean;
    case StyleRole::Warning:
        return theme_.warning;
    }

    return theme_.value;
}

bool stdout_is_terminal()
{
    return terminal_is_tty();
}

ColorMode parse_color_mode(std::string_view text)
{
    const auto mode = normalized(text);
    if (mode == "auto") {
        return ColorMode::Auto;
    }
    if (mode == "never" || mode == "none" || mode == "off" || mode == "false") {
        return ColorMode::Never;
    }
    if (mode == "always" || mode == "force" || mode == "on" || mode == "true") {
        return ColorMode::Always;
    }

    throw std::invalid_argument("unknown color mode");
}

GlyphMode parse_glyph_mode(std::string_view text)
{
    const auto mode = normalized(text);
    if (mode == "ascii") {
        return GlyphMode::Ascii;
    }
    if (mode == "utf8" || mode == "unicode") {
        return GlyphMode::Utf8;
    }

    throw std::invalid_argument("unknown glyph mode");
}

GlyphSet glyphs_for(GlyphMode mode)
{
    switch (mode) {
    case GlyphMode::Ascii:
        return GlyphSet::ascii();
    case GlyphMode::Utf8:
        return GlyphSet::utf8();
    }

    return GlyphSet::ascii();
}

} // namespace leakflow::render
