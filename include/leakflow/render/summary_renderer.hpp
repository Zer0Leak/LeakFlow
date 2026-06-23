#pragma once

#include "leakflow/core/summary_document.hpp"
#include "leakflow/render/terminal_style.hpp"

#include <string>

namespace leakflow::render {

struct SummaryRenderOptions {
    SummaryRenderOptions();

    ColorMode color_mode = ColorMode::Auto;
    GlyphMode glyph_mode = GlyphMode::Utf8;
    TerminalTheme theme;
};

class SummaryRenderer {
public:
    explicit SummaryRenderer(SummaryRenderOptions options = SummaryRenderOptions{});

    [[nodiscard]] std::string render(const SummaryDocument& document) const;

private:
    [[nodiscard]] std::string styled_value(const SummaryValue& value) const;
    void render_field(std::string& output,
        const std::string& prefix,
        const SummaryField& field,
        bool last) const;

    SummaryRenderOptions options_;
    TerminalStyler styler_;
    GlyphSet glyphs_;
};

[[nodiscard]] std::string render_summary_plain(const SummaryDocument& document, GlyphMode glyph_mode = GlyphMode::Ascii);

} // namespace leakflow::render
