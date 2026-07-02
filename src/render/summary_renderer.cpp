#include "leakflow/render/summary_renderer.hpp"

#include <sstream>
#include <utility>

namespace leakflow::render {
namespace {

[[nodiscard]] StyleRole style_role_for(SummaryValueRole role)
{
    switch (role) {
    case SummaryValueRole::Text:
        return StyleRole::Value;
    case SummaryValueRole::TypeName:
        return StyleRole::Type;
    case SummaryValueRole::CapsType:
        return StyleRole::Feature;
    case SummaryValueRole::MetadataKey:
        return StyleRole::Label;
    case SummaryValueRole::Path:
        return StyleRole::Path;
    case SummaryValueRole::Number:
        return StyleRole::Number;
    case SummaryValueRole::Size:
        return StyleRole::Size;
    case SummaryValueRole::Boolean:
        return StyleRole::Boolean;
    case SummaryValueRole::Warning:
        return StyleRole::Warning;
    }

    return StyleRole::Value;
}

} // namespace

SummaryRenderOptions::SummaryRenderOptions()
    : theme(TerminalTheme::leakflow())
{
}

SummaryRenderer::SummaryRenderer(SummaryRenderOptions options)
    : options_(std::move(options))
    , styler_(options_.color_mode, options_.theme)
    , glyphs_(glyphs_for(options_.glyph_mode))
{
}

std::string SummaryRenderer::render(const SummaryDocument& document) const
{
    std::string output;
    output += styler_.apply(StyleRole::Heading, document.title);
    output += '\n';

    for (std::size_t section_index = 0; section_index < document.sections.size(); ++section_index) {
        const auto& section = document.sections[section_index];
        const auto section_is_last = section_index + 1 == document.sections.size();

        output += styler_.apply(StyleRole::Tree, section_is_last ? glyphs_.leaf : glyphs_.branch);
        output += ' ';
        output += styler_.apply(StyleRole::Section, section.title);
        output += '\n';

        std::string field_prefix = section_is_last ? "   " : glyphs_.vertical + "  ";
        for (std::size_t field_index = 0; field_index < section.fields.size(); ++field_index) {
            render_field(output, field_prefix, section.fields[field_index], field_index + 1 == section.fields.size());
        }
    }

    return output;
}

std::string SummaryRenderer::styled_value(const SummaryValue& value) const
{
    return styler_.apply(style_role_for(value.role), value.text);
}

void SummaryRenderer::render_field(std::string& output,
    const std::string& prefix,
    const SummaryField& field,
    bool last) const
{
    output += prefix;
    output += styler_.apply(StyleRole::Tree, last ? glyphs_.leaf : glyphs_.branch);
    output += ' ';
    output += styler_.apply(StyleRole::Label, field.label);
    // A field with no value renders as a bare header label (no trailing '=').
    if (!field.value.text.empty()) {
        output += '=';
        output += styled_value(field.value);
    }
    output += '\n';

    const std::string child_prefix = prefix + (last ? "   " : glyphs_.vertical + "  ");
    for (std::size_t child_index = 0; child_index < field.children.size(); ++child_index) {
        render_field(output, child_prefix, field.children[child_index], child_index + 1 == field.children.size());
    }
}

std::string render_summary_plain(const SummaryDocument& document, GlyphMode glyph_mode)
{
    SummaryRenderOptions options;
    options.color_mode = ColorMode::Never;
    options.glyph_mode = glyph_mode;
    return SummaryRenderer(options).render(document);
}

} // namespace leakflow::render
