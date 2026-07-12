#include "leakflow/core/buffer.hpp"
#include "leakflow/render/summary_renderer.hpp"
#include "leakflow/render/terminal_style.hpp"

#include <iostream>
#include <memory>
#include <string>

namespace {

class TestPayload final : public leakflow::Payload {
public:
    [[nodiscard]] std::string type_name() const override
    {
        return "test/payload";
    }

    [[nodiscard]] std::string layout() const override
    {
        return "row/column";
    }

    void describe(leakflow::SummarySection& section, std::int64_t summary_level) const override
    {
        section.add_field("payload", type_name(), leakflow::SummaryValueRole::TypeName);
        section.add_field("rank", leakflow::summary_integer(2), leakflow::SummaryValueRole::Number);
        if (summary_level >= 2) {
            section.add_field("shape", "[3, 4]", leakflow::SummaryValueRole::Number);
        }
    }
};

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

} // namespace

int main()
{
    leakflow::Buffer buffer(leakflow::Caps("test/caps", {
        {"dtype", "float32"},
    }));
    buffer.set_metadata("file.path", "fixtures/input.npy");
    buffer.set_payload(std::make_shared<TestPayload>());

    const auto document = buffer.describe(3);
    if (!expect(document.title == "Buffer", "buffer summary document title changed")) {
        return 1;
    }
    if (!expect(document.sections.size() == 4, "buffer summary document section count changed")) {
        return 1;
    }
    if (!expect(document.sections[0].fields.size() == 2, "overview should only include buffer envelope fields")) {
        return 1;
    }
    if (!expect(document.sections[0].fields[0].label == "caps", "overview did not include caps first")) {
        return 1;
    }
    if (!expect(document.sections[0].fields[1].label == "metadata", "overview did not include metadata second")) {
        return 1;
    }

    const auto plain = leakflow::render::render_summary_plain(document, leakflow::render::GlyphMode::Ascii);
    if (!expect(plain.find("caps=test/caps; dtype=float32") != std::string::npos,
            "plain summary did not include caps")) {
        return 1;
    }
    if (!expect(plain.find("file.path=fixtures/input.npy") != std::string::npos,
            "plain summary did not include metadata value")) {
        return 1;
    }
    if (!expect(plain.find("payload=test/payload") != std::string::npos,
            "plain summary did not include payload type")) {
        return 1;
    }
    if (!expect(plain.find("shape=[3, 4]") != std::string::npos,
            "plain summary did not include payload detail")) {
        return 1;
    }

    leakflow::render::SummaryRenderOptions colored_options;
    colored_options.color_mode = leakflow::render::ColorMode::Always;
    colored_options.glyph_mode = leakflow::render::GlyphMode::Utf8;
    const auto colored = leakflow::render::SummaryRenderer(colored_options).render(document);
    if (!expect(colored.find("\033[") != std::string::npos, "forced color summary did not include ANSI codes")) {
        return 1;
    }
    if (!expect(colored.find("├─") != std::string::npos, "UTF-8 summary did not include branch glyphs")) {
        return 1;
    }

    if (!expect(leakflow::render::parse_color_mode("always") == leakflow::render::ColorMode::Always,
            "color mode parsing failed")) {
        return 1;
    }
    if (!expect(leakflow::render::parse_glyph_mode("utf8") == leakflow::render::GlyphMode::Utf8,
            "glyph mode parsing failed")) {
        return 1;
    }

    return 0;
}
