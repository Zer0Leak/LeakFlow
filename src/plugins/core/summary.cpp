#include "leakflow/plugins/core/summary.hpp"

#include "leakflow/render/summary_renderer.hpp"
#include "core_plugin_constants.hpp"
#include "property_helpers.hpp"

#include <cstdint>
#include <utility>

namespace leakflow::plugins::core {

ElementDescriptor Summary::descriptor()
{
    return {
        .type_name = "Summary",
        .klass = "PassThrough/Inspect/Summary",
        .purpose = "render a structured description of the current buffer",
        .input_pads = {
            Pad("sink", PadDirection::Input, Caps(core_pad_caps_type)),
        },
        .output_pads = {
            Pad("src", PadDirection::Output, Caps(core_pad_caps_type)),
        },
        .property_specs = {
            PropertySpec("metadata_key", std::string("summary"), "metadata key used for buffer summaries"),
            PropertySpec(
                "level",
                std::int64_t{1},
                "summary detail level",
                "level",
                IntRangeConstraint{0, 3}),
            PropertySpec(
                "color",
                std::string("auto"),
                "terminal color mode for rendered summaries",
                "",
                StringEnumConstraint{{"auto", "never", "always"}}),
            PropertySpec(
                "glyphs",
                std::string("utf8"),
                "glyph set used by rendered summaries",
                "",
                StringEnumConstraint{{"ascii", "utf8"}}),
            PropertySpec(
                "always_print",
                false,
                "render summaries even when global summary output is disabled"),
        },
        .keywords = {"summary", "inspect", "buffer"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "<metadata_key>",
                std::string(),
                "plain-text summary rendered by Summary; defaults to summary",
                {"summary=<plain-text buffer summary>"}),
        },
    };
}

Summary::Summary(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> Summary::process(std::optional<Buffer> input)
{
    if (!input) {
        last_summary_.clear();
        last_plain_summary_.clear();
        return std::nullopt;
    }

    if (!leakflow::log::summaries_enabled() && !bool_property_or(*this, "always_print", false)) {
        last_summary_.clear();
        last_plain_summary_.clear();
        leakflow::log::write(make_log_record(log::LogLevel::Debug, "element", "suppressed summary rendering"));
        return input;
    }

    const auto level = int_property_or(*this, "level", 1);
    auto document = input->describe(level);
    document.title += "@";
    document.title += name();

    render::SummaryRenderOptions render_options;
    render_options.color_mode = render::parse_color_mode(string_property_or(*this, "color", "auto"));
    render_options.glyph_mode = render::parse_glyph_mode(string_property_or(*this, "glyphs", "utf8"));

    last_plain_summary_ = render::render_summary_plain(document, render_options.glyph_mode);
    last_summary_ = render::SummaryRenderer(render_options).render(document);

    input->set_metadata(string_property_or(*this, "metadata_key", "summary"), last_plain_summary_);

    auto record = make_log_record(log::LogLevel::Debug, "element", "rendered summary");
    record.fields.emplace("level", std::to_string(level));
    leakflow::log::write(std::move(record));
    return input;
}

const std::string& Summary::last_summary() const
{
    return last_summary_;
}

const std::string& Summary::last_plain_summary() const
{
    return last_plain_summary_;
}

} // namespace leakflow::plugins::core
