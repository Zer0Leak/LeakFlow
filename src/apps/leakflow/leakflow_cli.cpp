#include "leakflow_cli.hpp"

#include "leakflow/core/caps.hpp"
#include "leakflow/core/descriptor_registry.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/log/logger.hpp"
#include "leakflow/plot/pipeline_graph.hpp"
#include "leakflow/plugins/base/base_elements.hpp"
#include "leakflow/plugins/base/descriptor_catalog.hpp"
#include "leakflow/plugins/core/buffer_summary.hpp"
#include "leakflow/plugins/core/core_elements.hpp"
#include "leakflow/plugins/core/descriptor_catalog.hpp"
#include "leakflow/plugins/crypto/crypto_elements.hpp"
#include "leakflow/plugins/crypto/descriptor_catalog.hpp"
#include "leakflow/plugins/extras/descriptor_catalog.hpp"
#include "leakflow/plugins/extras/extras_elements.hpp"
#include "leakflow/plugins/plot/descriptor_catalog.hpp"
#include "leakflow/plugins/plot/plot_elements.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace leakflow::cli {
namespace {

namespace core = leakflow::plugins::core;
namespace base = leakflow::plugins::base;
namespace crypto_plugin = leakflow::plugins::crypto;
namespace extras = leakflow::plugins::extras;
namespace plot_plugin = leakflow::plugins::plot;

struct Endpoint {
    std::shared_ptr<Element> element;
    std::optional<std::string> pad_name;
    bool pad_is_template = false;
};

struct PadCapsAnnotation {
    std::string element_name;
    std::string pad_name;
    Caps caps;
};

[[nodiscard]] DescriptorRegistry linked_descriptor_registry() {
    DescriptorRegistry registry;
    core::register_plugin_descriptors(registry);
    base::register_plugin_descriptors(registry);
    extras::register_plugin_descriptors(registry);
    crypto_plugin::register_plugin_descriptors(registry);
    plot_plugin::register_plugin_descriptors(registry);
    return registry;
}

[[nodiscard]] ElementFactoryRegistry
linked_element_factory_registry(std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime) {
    ElementFactoryRegistry registry;
    core::register_element_factories(registry);
    base::register_element_factories(registry);
    extras::register_element_factories(registry);
    crypto_plugin::register_element_factories(registry);
    plot_plugin::register_element_factories(registry, std::move(plot_runtime));
    return registry;
}

[[nodiscard]] const DescriptorRegistry &descriptor_registry() {
    static const auto registry = linked_descriptor_registry();
    return registry;
}

[[nodiscard]] std::string_view trim_view(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] std::string trim_to_string(std::string_view text) {
    const auto trimmed = trim_view(text);
    return std::string(trimmed);
}

[[nodiscard]] bool is_identifier_character(char character) {
    return std::isalnum(static_cast<unsigned char>(character)) != 0 || character == '_' || character == '-';
}

[[nodiscard]] bool is_pad_selector_character(char character) {
    return is_identifier_character(character) || character == '%';
}

[[nodiscard]] std::string parse_identifier(std::string_view text, std::size_t &index) {
    const auto begin = index;
    while (index < text.size() && is_identifier_character(text[index])) {
        ++index;
    }

    if (begin == index) {
        throw std::invalid_argument("expected identifier");
    }

    return std::string(text.substr(begin, index - begin));
}

[[nodiscard]] std::string parse_pad_selector(std::string_view text, std::size_t &index) {
    const auto begin = index;
    while (index < text.size() && is_pad_selector_character(text[index])) {
        ++index;
    }

    if (begin == index) {
        throw std::invalid_argument("expected pad selector");
    }

    return std::string(text.substr(begin, index - begin));
}

void skip_spaces(std::string_view text, std::size_t &index) {
    while (index < text.size() && std::isspace(static_cast<unsigned char>(text[index])) != 0) {
        ++index;
    }
}

[[nodiscard]] std::string normalize_line_continuations(std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());

    bool in_quote = false;
    bool escaped = false;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const auto character = text[index];

        if (escaped) {
            normalized.push_back(character);
            escaped = false;
            continue;
        }

        if (in_quote) {
            normalized.push_back(character);
            if (character == '\\') {
                escaped = true;
                continue;
            }
            if (character == '"') {
                in_quote = false;
            }
            continue;
        }

        if (character == '"') {
            normalized.push_back(character);
            in_quote = true;
            continue;
        }

        if (character == '\\') {
            auto cursor = index + 1;
            while (cursor < text.size() && (text[cursor] == ' ' || text[cursor] == '\t')) {
                ++cursor;
            }

            if (cursor < text.size() && (text[cursor] == '\n' || text[cursor] == '\r')) {
                if (text[cursor] == '\r' && cursor + 1 < text.size() && text[cursor + 1] == '\n') {
                    ++cursor;
                }
                normalized.push_back(' ');
                index = cursor;
                continue;
            }
        }

        normalized.push_back(character);
    }

    return normalized;
}

[[nodiscard]] std::string escaped_token_char(char character) {
    switch (character) {
    case '\n':
        return "\\n";
    case '\r':
        return "\\r";
    case '\t':
        return "\\t";
    case '\\':
        return "\\\\";
    case '\'':
        return "\\'";
    default:
        return std::string(1, character);
    }
}

[[nodiscard]] std::string token_preview(std::string_view text, std::size_t index) {
    if (index >= text.size()) {
        return "<end>";
    }

    return "'" + escaped_token_char(text[index]) + "'";
}

[[nodiscard]] std::string creation_label(std::string_view type_name, const std::optional<std::string> &explicit_name) {
    if (!explicit_name) {
        return std::string(type_name);
    }

    return std::string(type_name) + "@" + *explicit_name;
}

[[nodiscard]] std::string reference_label(std::string_view element_name, const std::optional<std::string> &pad_name) {
    auto label = "@" + std::string(element_name);
    if (pad_name) {
        label += ".";
        label += *pad_name;
    }

    return label;
}

[[nodiscard]] std::size_t find_matching(std::string_view text, std::size_t open_index, char open, char close) {
    int depth = 0;
    bool in_quote = false;
    bool escaped = false;

    for (std::size_t index = open_index; index < text.size(); ++index) {
        const auto character = text[index];

        if (escaped) {
            escaped = false;
            continue;
        }

        if (in_quote) {
            if (character == '\\') {
                escaped = true;
                continue;
            }
            if (character == '"') {
                in_quote = false;
            }
            continue;
        }

        if (character == '"') {
            in_quote = true;
            continue;
        }

        if (character == open) {
            ++depth;
            continue;
        }
        if (character == close) {
            --depth;
            if (depth == 0) {
                return index;
            }
        }
    }

    throw std::invalid_argument("unterminated grouped expression");
}

[[nodiscard]] std::vector<std::string_view> split_top_level(std::string_view text, char delimiter) {
    std::vector<std::string_view> parts;
    std::size_t begin = 0;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    bool in_quote = false;
    bool escaped = false;

    for (std::size_t index = 0; index < text.size(); ++index) {
        const auto character = text[index];

        if (escaped) {
            escaped = false;
            continue;
        }

        if (in_quote) {
            if (character == '\\') {
                escaped = true;
                continue;
            }
            if (character == '"') {
                in_quote = false;
            }
            continue;
        }

        if (character == '"') {
            in_quote = true;
            continue;
        }
        if (character == '(') {
            ++paren_depth;
            continue;
        }
        if (character == ')') {
            --paren_depth;
            if (paren_depth < 0) {
                throw std::invalid_argument("unexpected ')'");
            }
            continue;
        }
        if (character == '[') {
            ++bracket_depth;
            continue;
        }
        if (character == ']') {
            --bracket_depth;
            if (bracket_depth < 0) {
                throw std::invalid_argument("unexpected ']'");
            }
            continue;
        }
        if (character == '{') {
            ++brace_depth;
            continue;
        }
        if (character == '}') {
            --brace_depth;
            if (brace_depth < 0) {
                throw std::invalid_argument("unexpected '}'");
            }
            continue;
        }

        if (character == delimiter && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
            parts.push_back(text.substr(begin, index - begin));
            begin = index + 1;
        }
    }

    if (paren_depth != 0 || bracket_depth != 0 || brace_depth != 0 || in_quote) {
        throw std::invalid_argument("unterminated expression");
    }

    parts.push_back(text.substr(begin));
    return parts;
}

[[nodiscard]] std::string parse_string_token(std::string_view text) {
    const auto trimmed = trim_view(text);
    if (trimmed.empty()) {
        return {};
    }

    if (trimmed.front() != '"') {
        return std::string(trimmed);
    }

    if (trimmed.size() < 2 || trimmed.back() != '"') {
        throw std::invalid_argument("unterminated string property value");
    }

    std::string value;
    value.reserve(trimmed.size() - 2);

    bool escaped = false;
    for (std::size_t index = 1; index + 1 < trimmed.size(); ++index) {
        const auto character = trimmed[index];

        if (escaped) {
            switch (character) {
            case 'n':
                value.push_back('\n');
                break;
            case 't':
                value.push_back('\t');
                break;
            case '"':
            case '\\':
                value.push_back(character);
                break;
            default:
                value.push_back(character);
                break;
            }
            escaped = false;
            continue;
        }

        if (character == '\\') {
            escaped = true;
            continue;
        }

        value.push_back(character);
    }

    if (escaped) {
        throw std::invalid_argument("unterminated string escape");
    }

    return value;
}

[[nodiscard]] std::string lower_string(std::string_view text) {
    std::string lowered;
    lowered.reserve(text.size());
    for (const auto character : text) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    return lowered;
}

[[nodiscard]] std::int64_t parse_integer(std::string_view text) {
    const auto trimmed = trim_view(text);
    std::int64_t value = 0;
    const auto result = std::from_chars(trimmed.data(), trimmed.data() + trimmed.size(), value);
    if (result.ec != std::errc{} || result.ptr != trimmed.data() + trimmed.size()) {
        throw std::invalid_argument("invalid integer value");
    }
    return value;
}

[[nodiscard]] double parse_double(std::string_view text) {
    const auto trimmed = trim_to_string(text);
    std::size_t consumed = 0;
    const double value = std::stod(trimmed, &consumed);
    if (consumed != trimmed.size()) {
        throw std::invalid_argument("invalid double value");
    }
    return value;
}

[[nodiscard]] std::vector<std::string_view> split_list_items(std::string_view text) {
    const auto trimmed = trim_view(text);
    if (trimmed.size() < 2 || trimmed.front() != '[' || trimmed.back() != ']') {
        throw std::invalid_argument("list property value must use []");
    }

    const auto inner = trim_view(trimmed.substr(1, trimmed.size() - 2));
    if (inner.empty()) {
        return {};
    }

    return split_top_level(inner, ',');
}

[[nodiscard]] std::size_t find_interval_separator(std::string_view text) {
    bool in_quote = false;
    bool escaped = false;
    int bracket_depth = 0;

    for (std::size_t index = 0; index + 1 < text.size(); ++index) {
        const auto character = text[index];

        if (escaped) {
            escaped = false;
            continue;
        }

        if (in_quote) {
            if (character == '\\') {
                escaped = true;
                continue;
            }
            if (character == '"') {
                in_quote = false;
            }
            continue;
        }

        if (character == '"') {
            in_quote = true;
            continue;
        }
        if (character == '[') {
            ++bracket_depth;
            continue;
        }
        if (character == ']') {
            --bracket_depth;
            continue;
        }

        if (bracket_depth == 0 && character == '.' && text[index + 1] == '.') {
            return index;
        }
    }

    throw std::invalid_argument("interval property value must use '..'");
}

[[nodiscard]] const PropertySpec &find_property_spec(const Element &element, std::string_view name) {
    for (const auto &spec : element.property_specs()) {
        if (spec.name == name) {
            return spec;
        }
    }

    throw std::invalid_argument("unknown property");
}

[[nodiscard]] const Pad *find_pad(const Element &element, std::string_view name) {
    for (const auto &pad : element.input_pads()) {
        if (pad.name() == name) {
            return &pad;
        }
    }
    for (const auto &pad : element.output_pads()) {
        if (pad.name() == name) {
            return &pad;
        }
    }

    return nullptr;
}

[[nodiscard]] bool has_unsigned_pad_placeholder(std::string_view pad_selector) {
    return pad_selector.find("%u") != std::string_view::npos || pad_selector.find("%U") != std::string_view::npos;
}

[[nodiscard]] bool output_pad_template_matches_any(const Element &element, std::string_view pad_template) {
    for (const auto &declared_template : element.pad_templates()) {
        if (declared_template.direction() == PadDirection::Output && declared_template.name() == pad_template) {
            return true;
        }
    }

    for (const auto &pad : element.output_pads()) {
        if (metadata_pad_template_matches(pad_template, pad.name())) {
            return true;
        }
    }

    return false;
}

[[nodiscard]] const Pad &infer_output_pad(const Element &element) {
    if (element.output_pads().empty()) {
        throw std::invalid_argument("element has no output pad");
    }
    if (element.output_pads().size() != 1) {
        throw std::invalid_argument("ambiguous output pad; use @element.pad");
    }

    return element.output_pads().front();
}

[[nodiscard]] const Pad &infer_input_pad(const Element &element) {
    if (element.input_pads().empty()) {
        throw std::invalid_argument("element has no input pad");
    }
    if (element.input_pads().size() != 1) {
        throw std::invalid_argument("ambiguous input pad; use @element.pad");
    }

    return element.input_pads().front();
}

[[nodiscard]] const Pad &infer_input_pad(const Element &element, const Caps &source_caps) {
    if (element.input_pads().empty()) {
        throw std::invalid_argument("element has no input pad");
    }
    if (element.input_pads().size() == 1) {
        return element.input_pads().front();
    }

    const Pad *match = nullptr;
    for (const auto &pad : element.input_pads()) {
        if (!caps_are_compatible(source_caps, pad.caps())) {
            continue;
        }
        if (match != nullptr) {
            throw std::invalid_argument("ambiguous input pad; use @element.pad");
        }

        match = &pad;
    }

    if (match == nullptr) {
        throw std::invalid_argument("no compatible input pad; use @element.pad");
    }

    return *match;
}

[[nodiscard]] std::string canonical_type_name(std::string_view text) {
    if (const auto descriptor = descriptor_registry().find_element(text)) {
        return descriptor->element->type_name;
    }

    throw std::invalid_argument("unknown element type");
}

[[nodiscard]] std::shared_ptr<Element> make_element(std::string_view type_name, std::string name,
                                                    const ElementFactoryRegistry &element_factories) {
    return element_factories.create(type_name, std::move(name));
}

enum class NamePropertyMode {
    Allow,
    Reject,
};

[[nodiscard]] std::optional<std::string> extract_name_property(std::string_view properties) {
    const auto trimmed = trim_view(properties);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    std::optional<std::string> name;
    for (const auto assignment : split_top_level(trimmed, ',')) {
        const auto part = trim_view(assignment);
        const auto equals = part.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument("property assignment must use key=value");
        }

        const auto property_name = trim_view(part.substr(0, equals));
        if (property_name.empty()) {
            throw std::invalid_argument("property name cannot be empty");
        }
        if (property_name != "name") {
            continue;
        }
        if (name) {
            throw std::invalid_argument("duplicate name property");
        }

        name = parse_string_token(part.substr(equals + 1));
    }

    return name;
}

void apply_properties(Element &element, std::string_view properties, NamePropertyMode name_property_mode) {
    const auto trimmed = trim_view(properties);
    if (trimmed.empty()) {
        return;
    }

    for (const auto assignment : split_top_level(trimmed, ',')) {
        const auto part = trim_view(assignment);
        const auto equals = part.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument("property assignment must use key=value");
        }

        const auto property_name = trim_view(part.substr(0, equals));
        const auto property_text = part.substr(equals + 1);
        if (property_name.empty()) {
            throw std::invalid_argument("property name cannot be empty");
        }
        if (property_name == "name" && name_property_mode == NamePropertyMode::Reject) {
            throw std::invalid_argument("element name cannot be changed after creation");
        }

        const auto &spec = find_property_spec(element, property_name);
        auto value = parse_property_value(spec, property_text);
        if (property_name == "name" && name_property_mode == NamePropertyMode::Allow && element.name_locked()) {
            const auto *name_value = std::get_if<std::string>(&value);
            if (name_value != nullptr && *name_value == element.name()) {
                continue;
            }
        }

        element.set_property(property_name, std::move(value));
    }
}

[[nodiscard]] std::map<std::string, std::string> parse_string_map_annotation(std::string_view text,
                                                                             std::string_view label) {
    std::map<std::string, std::string> values;

    for (const auto part_text : split_top_level(text, ';')) {
        const auto part = trim_view(part_text);
        if (part.empty()) {
            continue;
        }

        const auto equals = part.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument(std::string(label) + " annotation entries must use key=value");
        }

        const auto key = trim_to_string(part.substr(0, equals));
        const auto value = parse_string_token(part.substr(equals + 1));
        if (key.empty()) {
            throw std::invalid_argument(std::string(label) + " annotation key cannot be empty");
        }
        if (values.contains(key)) {
            throw std::invalid_argument("duplicate " + std::string(label) + " annotation key");
        }

        values.emplace(key, value);
    }

    return values;
}

// SIGINT -> cooperative stop bridge (live phase, S11.8). The signal handler only
// sets an async-signal-safe flag; a watcher thread observes it and requests the
// stop_source (request_stop is not itself signal-safe). The scope restores the
// previous SIGINT disposition on destruction. Use one InterruptScope around a
// headless live run so Ctrl+C drains the pump gracefully instead of hard-killing.
volatile std::sig_atomic_t g_interrupt_requested = 0;

extern "C" void leakflow_handle_sigint(int) { g_interrupt_requested = 1; }

class InterruptScope {
public:
    InterruptScope() {
        g_interrupt_requested = 0;
        previous_handler_ = std::signal(SIGINT, leakflow_handle_sigint);
        watcher_ = std::jthread([this](std::stop_token st) {
            using namespace std::chrono_literals;
            while (!st.stop_requested()) {
                if (g_interrupt_requested != 0) {
                    source_.request_stop();
                    return;
                }
                std::this_thread::sleep_for(50ms);
            }
        });
    }

    InterruptScope(const InterruptScope &) = delete;
    InterruptScope &operator=(const InterruptScope &) = delete;

    ~InterruptScope() {
        watcher_.request_stop(); // jthread joins in its destructor
        if (previous_handler_ != SIG_ERR) {
            (void)std::signal(SIGINT, previous_handler_);
        }
    }

    [[nodiscard]] std::stop_token token() const { return source_.get_token(); }

private:
    std::stop_source source_;
    std::jthread watcher_;
    void (*previous_handler_)(int) = nullptr;
};

class ExpressionRunner {
public:
    explicit ExpressionRunner(std::ostream &output)
        : output_(output), plot_runtime_(std::make_shared<leakflow::plot::PlotRuntime>()),
          element_factories_(linked_element_factory_registry(plot_runtime_)) {}

    ExpressionRunner(std::ostream &output, ElementFactoryRegistry element_factories,
                     std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime)
        : output_(output), plot_runtime_(std::move(plot_runtime)), element_factories_(std::move(element_factories)) {
        if (!plot_runtime_) {
            plot_runtime_ = std::make_shared<leakflow::plot::PlotRuntime>();
        }
    }

    Pipeline build(std::string_view expression) {
        const auto normalized_expression = normalize_line_continuations(expression);
        parse_expression(normalized_expression);
        return std::move(pipeline_);
    }

    int run(std::string_view expression, RunExpressionOptions options) {
        leakflow::log::write(log::LogRecord{
            .level = log::LogLevel::Debug,
            .component = "cli",
            .message = "running pipeline expression",
        });
        const auto normalized_expression = normalize_line_continuations(expression);
        parse_expression(normalized_expression);

        if (options.graph) {
            return run_with_graph(options.auto_start);
        }

        // Headless front door (Phase 25): execute through a PipelineSession so the
        // session is the single execution path. run_once does start -> sweep ->
        // stop. See docs/design/pipeline_controller.md.
        leakflow::PipelineSession session(std::move(pipeline_));
        // Ctrl+C requests a cooperative stop: a paced live pump drains gracefully
        // (and an interruptible source wait ends mid-trace) instead of being
        // hard-killed (S11.8). Offline runs are unaffected (single sweep).
        InterruptScope interrupt;
        session.set_stop_token(interrupt.token());
        auto result = session.run_once();
        print_outputs(result);
        run_plot_runtime_if_needed();
        leakflow::log::write(log::LogRecord{
            .level = log::LogLevel::Debug,
            .component = "cli",
            .message = "finished pipeline expression",
        });
        return 0;
    }

private:
    int run_with_graph(bool auto_start) {
        leakflow::plot::PlotLoopOptions options;
        options.window_title = "LeakFlow Pipeline Graph";
        options.auto_start = auto_start;
        // Phase 25: drive the graph through a PipelineSession so live property
        // edits apply at safe points with downstream rerun.
        leakflow::PipelineSession session(std::move(pipeline_));
        const auto result = leakflow::plot::run_pipeline_graph_until_closed(session, *plot_runtime_, options);

        print_outputs(result);
        leakflow::log::write(log::LogRecord{
            .level = log::LogLevel::Debug,
            .component = "cli",
            .message = "finished graph pipeline expression",
        });
        return 0;
    }

    void parse_expression(std::string_view expression) {
        for (const auto statement : split_top_level(expression, ';')) {
            const auto trimmed = trim_view(statement);
            if (trimmed.empty()) {
                continue;
            }
            parse_statement(trimmed);
        }

        if (elements_.empty()) {
            throw std::invalid_argument("pipeline expression did not create any elements");
        }
    }

    void parse_statement(std::string_view statement) {
        const auto parts = split_top_level(statement, '!');
        if (parts.empty()) {
            return;
        }

        std::vector<Endpoint> endpoints;
        endpoints.reserve(parts.size());
        for (const auto part : parts) {
            const auto trimmed = trim_view(part);
            if (trimmed.empty()) {
                throw std::invalid_argument("empty link endpoint");
            }
            endpoints.push_back(parse_endpoint(trimmed));
        }

        for (std::size_t index = 1; index < endpoints.size(); ++index) {
            add_link(endpoints[index - 1], endpoints[index]);
        }
    }

    [[nodiscard]] Endpoint parse_endpoint(std::string_view text) {
        if (text.front() == '@') {
            return parse_reference(text);
        }

        return parse_creation(text);
    }

    [[nodiscard]] Endpoint parse_creation(std::string_view text) {
        std::size_t index = 0;
        const auto type_name = canonical_type_name(parse_identifier(text, index));
        skip_spaces(text, index);

        std::optional<std::string> explicit_name;
        if (index < text.size() && text[index] == '@') {
            ++index;
            explicit_name = parse_identifier(text, index);
            skip_spaces(text, index);
        }

        std::string property_text;
        if (index < text.size() && text[index] == '(') {
            const auto close = find_matching(text, index, '(', ')');
            property_text = std::string(text.substr(index + 1, close - index - 1));
            index = close + 1;
            skip_spaces(text, index);
        }

        const auto name_property = extract_name_property(property_text);
        if (explicit_name && name_property && *explicit_name != *name_property) {
            throw std::invalid_argument("conflicting element instance names");
        }
        const auto creation_name = name_property ? name_property : explicit_name;
        const auto element = create_element(type_name, creation_name);
        apply_properties(*element, property_text, NamePropertyMode::Allow);

        // Optional pad selector on creation: `Type@name.pad` addresses a specific
        // (possibly on-request) pad of the just-created element -- e.g. a fan-in join
        // `... ! Sync@s.in_0` or a fan-out `Tee@t.src_0 ! ...`. The same instance is
        // referenced later with `@name.other_pad` (see parse_reference). Note the
        // join must be declared after its inputs (source-before-sink add order).
        std::optional<std::string> pad_name;
        bool pad_is_template = false;
        if (index < text.size() && text[index] == '.') {
            ++index;
            pad_name = parse_pad_selector(text, index);
            pad_is_template = has_unsigned_pad_placeholder(*pad_name);
            skip_spaces(text, index);
        }

        while (index < text.size()) {
            if (text[index] == '[') {
                const auto close = find_matching(text, index, '[', ']');
                add_pad_caps_annotation(element, pad_name,
                                        parse_caps_annotation(text.substr(index + 1, close - index - 1)));
                index = close + 1;
                skip_spaces(text, index);
                continue;
            }
            if (text[index] == '{') {
                const auto close = find_matching(text, index, '{', '}');
                add_pad_metadata_annotation(element, pad_name, pad_is_template,
                                            parse_metadata_annotation(text.substr(index + 1, close - index - 1)));
                index = close + 1;
                skip_spaces(text, index);
                continue;
            }

            throw std::invalid_argument("unexpected token " + token_preview(text, index) + " after element creation " +
                                        creation_label(type_name, explicit_name) +
                                        "; expected '.', '(', '[', '{', '!', or ';'");
        }

        if (pad_name && !pad_is_template) {
            validate_pad_exists(*element, *pad_name);
        }

        return Endpoint{element, std::move(pad_name), pad_is_template};
    }

    [[nodiscard]] Endpoint parse_reference(std::string_view text) {
        std::size_t index = 1;
        const auto element_name = parse_identifier(text, index);
        const auto element = find_named_element(element_name);
        skip_spaces(text, index);

        std::optional<std::string> pad_name;
        bool pad_is_template = false;
        if (index < text.size() && text[index] == '.') {
            ++index;
            pad_name = parse_pad_selector(text, index);
            pad_is_template = has_unsigned_pad_placeholder(*pad_name);
            skip_spaces(text, index);
        }

        if (index < text.size() && text[index] == '(') {
            if (pad_name) {
                throw std::invalid_argument("pad references cannot set element properties");
            }
            const auto close = find_matching(text, index, '(', ')');
            apply_properties(*element, text.substr(index + 1, close - index - 1), NamePropertyMode::Reject);
            index = close + 1;
            skip_spaces(text, index);
        }

        while (index < text.size()) {
            if (text[index] == '[') {
                const auto close = find_matching(text, index, '[', ']');
                add_pad_caps_annotation(element, pad_name,
                                        parse_caps_annotation(text.substr(index + 1, close - index - 1)));
                index = close + 1;
                skip_spaces(text, index);
                continue;
            }
            if (text[index] == '{') {
                const auto close = find_matching(text, index, '{', '}');
                add_pad_metadata_annotation(element, pad_name, pad_is_template,
                                            parse_metadata_annotation(text.substr(index + 1, close - index - 1)));
                index = close + 1;
                skip_spaces(text, index);
                continue;
            }

            throw std::invalid_argument("unexpected token " + token_preview(text, index) + " after element reference " +
                                        reference_label(element_name, pad_name) +
                                        "; expected '(', '[', '{', '!', or ';'");
        }

        if (pad_name && !pad_is_template) {
            validate_pad_exists(*element, *pad_name);
        }

        return Endpoint{element, std::move(pad_name), pad_is_template};
    }

    [[nodiscard]] std::shared_ptr<Element> create_element(std::string_view type_name,
                                                          const std::optional<std::string> &explicit_name) {
        const auto name = explicit_name ? *explicit_name : unique_generated_name(type_name);
        if (name.empty()) {
            throw std::invalid_argument("element name cannot be empty");
        }

        auto element = make_element(type_name, name, element_factories_);
        pipeline_.add(element);
        elements_.push_back(element);

        auto record = element->make_log_record(log::LogLevel::Debug, "cli", "created element");
        record.fields.emplace("explicit_name", explicit_name ? "true" : "false");
        leakflow::log::write(std::move(record));
        return element;
    }

    [[nodiscard]] std::string unique_generated_name(std::string_view type_name) const {
        const auto base = element_default_name_prefix(type_name);
        for (std::size_t index = 0;; ++index) {
            auto candidate = base + std::to_string(index);
            if (pipeline_.find_element(candidate) == nullptr) {
                return candidate;
            }
        }
    }

    [[nodiscard]] std::shared_ptr<Element> find_named_element(std::string_view name) const {
        auto found = pipeline_.find_element(name);
        if (!found) {
            throw std::invalid_argument("unknown named element");
        }

        return found;
    }

    void validate_pad_exists(const Element &element, std::string_view pad_name) const {
        if (find_pad(element, pad_name) == nullptr && !element.can_request_pad(pad_name)) {
            throw std::invalid_argument("unknown pad");
        }
    }

    void validate_output_pad_template(const Element &element, std::string_view pad_template) const {
        if (!has_unsigned_pad_placeholder(pad_template)) {
            throw std::invalid_argument("pad template selector must contain %u");
        }
        if (!output_pad_template_matches_any(element, pad_template)) {
            throw std::invalid_argument("pad template selector matches no output pads");
        }
    }

    [[nodiscard]] std::string annotation_pad_name(const Element &element,
                                                  const std::optional<std::string> &pad_name) const {
        if (pad_name) {
            validate_pad_exists(element, *pad_name);
            return *pad_name;
        }

        return infer_output_pad(element).name();
    }

    void add_pad_caps_annotation(const std::shared_ptr<Element> &element, const std::optional<std::string> &pad_name,
                                 Caps caps) {
        const auto resolved_pad_name = annotation_pad_name(*element, pad_name);
        // TODO: Apply parsed pad caps to element pads or link validation once
        // LeakFlow has mutable pad declarations or a real caps negotiation model.
        pad_caps_annotations_.push_back(PadCapsAnnotation{element->name(), resolved_pad_name, std::move(caps)});
    }

    void add_pad_metadata_annotation(const std::shared_ptr<Element> &element,
                                     const std::optional<std::string> &pad_name, bool pad_is_template,
                                     std::map<std::string, std::string> metadata) {
        if (!pad_name) {
            pipeline_.add_output_metadata_annotation(element, std::move(metadata));
            return;
        }

        if (pad_is_template) {
            validate_output_pad_template(*element, *pad_name);
            pipeline_.add_output_metadata_annotation_for_pad_template(element, *pad_name, std::move(metadata));
            return;
        }

        validate_pad_exists(*element, *pad_name);
        pipeline_.add_output_metadata_annotation(element, *pad_name, std::move(metadata));
    }

    [[nodiscard]] std::string output_pad_name(const Endpoint &endpoint) const {
        if (endpoint.pad_name) {
            if (endpoint.pad_is_template) {
                throw std::invalid_argument("pad template selectors cannot be used as link endpoints");
            }
            bool found_output = false;
            for (const auto &pad : endpoint.element->output_pads()) {
                if (pad.name() == *endpoint.pad_name) {
                    found_output = true;
                    break;
                }
            }
            if (!found_output) {
                found_output = endpoint.element->request_output_pad(*endpoint.pad_name);
            }
            if (!found_output) {
                throw std::invalid_argument("source endpoint is not an output pad");
            }
            return *endpoint.pad_name;
        }

        return infer_output_pad(*endpoint.element).name();
    }

    [[nodiscard]] std::string input_pad_name(const Endpoint &endpoint, const Caps &source_caps) const {
        if (endpoint.pad_name) {
            if (endpoint.pad_is_template) {
                throw std::invalid_argument("pad template selectors cannot be used as link endpoints");
            }
            bool found_input = false;
            for (const auto &pad : endpoint.element->input_pads()) {
                if (pad.name() == *endpoint.pad_name) {
                    found_input = true;
                    break;
                }
            }
            if (!found_input) {
                found_input = endpoint.element->request_input_pad(*endpoint.pad_name);
            }
            if (!found_input) {
                throw std::invalid_argument("sink endpoint is not an input pad");
            }
            return *endpoint.pad_name;
        }

        return infer_input_pad(*endpoint.element, source_caps).name();
    }

    void add_link(const Endpoint &source, const Endpoint &sink) {
        const auto source_pad = output_pad_name(source);
        const auto source_caps = pipeline_.source_caps(*source.element, source_pad);
        const auto sink_pad = input_pad_name(sink, source_caps);
        pipeline_.link(source.element, source_pad, sink.element, sink_pad);
    }

    void print_outputs(const std::optional<Buffer> &result) {
        for (const auto &element : elements_) {
            if (const auto summary = std::dynamic_pointer_cast<core::Summary>(element)) {
                if (!summary->last_summary().empty()) {
                    output_ << summary->last_summary() << '\n';
                }
            }
        }

        (void)result;
    }

    void run_plot_runtime_if_needed() {
        if (!plot_runtime_->has_sessions()) {
            return;
        }

        leakflow::log::write(log::LogRecord{
            .level = log::LogLevel::Debug,
            .component = "cli",
            .message = "entering plot runtime loop",
        });
        leakflow::plot::run_until_closed(*plot_runtime_);
    }

    std::ostream &output_;
    Pipeline pipeline_;
    std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime_;
    ElementFactoryRegistry element_factories_;
    std::vector<std::shared_ptr<Element>> elements_;
    std::vector<PadCapsAnnotation> pad_caps_annotations_;
};

} // namespace

std::string normalized_identifier(std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());

    for (const auto character : text) {
        if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
        }
    }

    return normalized;
}

PropertyValue parse_property_value(const PropertySpec &spec, std::string_view text) {
    const auto &default_value = spec.default_value;

    // Optional properties accept an explicit null: null/nil/none/auto/empty.
    if (spec.optional) {
        const auto token = lower_string(trim_view(text));
        if (token.empty() || token == "null" || token == "nil" || token == "none" || token == "auto") {
            return PropertyValue{std::monostate{}};
        }
    }

    PropertyValue value = std::visit(
        [text](const auto &typed_default) -> PropertyValue {
            using Value = std::decay_t<decltype(typed_default)>;

            if constexpr (std::is_same_v<Value, bool>) {
                const auto lowered = lower_string(trim_view(text));
                if (lowered == "true" || lowered == "1") {
                    return true;
                }
                if (lowered == "false" || lowered == "0") {
                    return false;
                }
                throw std::invalid_argument("invalid bool property value");
            } else if constexpr (std::is_same_v<Value, std::int64_t>) {
                return parse_integer(text);
            } else if constexpr (std::is_same_v<Value, double>) {
                return parse_double(text);
            } else if constexpr (std::is_same_v<Value, std::string>) {
                return parse_string_token(text);
            } else if constexpr (std::is_same_v<Value, IntInterval>) {
                const auto separator = find_interval_separator(text);
                return IntInterval{
                    parse_integer(text.substr(0, separator)),
                    parse_integer(text.substr(separator + 2)),
                };
            } else if constexpr (std::is_same_v<Value, DoubleInterval>) {
                const auto separator = find_interval_separator(text);
                return DoubleInterval{
                    parse_double(text.substr(0, separator)),
                    parse_double(text.substr(separator + 2)),
                };
            } else if constexpr (std::is_same_v<Value, IntList>) {
                IntList values;
                for (const auto item : split_list_items(text)) {
                    values.push_back(parse_integer(item));
                }
                return values;
            } else if constexpr (std::is_same_v<Value, DoubleList>) {
                DoubleList values;
                for (const auto item : split_list_items(text)) {
                    values.push_back(parse_double(item));
                }
                return values;
            } else if constexpr (std::is_same_v<Value, StringList>) {
                StringList values;
                for (const auto item : split_list_items(text)) {
                    values.push_back(parse_string_token(item));
                }
                return values;
            } else if constexpr (std::is_same_v<Value, Color>) {
                auto color = parse_color(parse_string_token(text));
                if (!color) {
                    throw std::invalid_argument("invalid color property value");
                }
                return *color;
            } else if constexpr (std::is_same_v<Value, std::monostate>) {
                return std::monostate{}; // unreachable: default_value is never null
            }
        },
        default_value);

    validate_property_value(spec, value);
    return value;
}

Caps parse_caps_annotation(std::string_view text) {
    std::optional<std::string> caps_type;
    Caps::Params params;

    for (const auto part_text : split_top_level(text, ';')) {
        const auto part = trim_view(part_text);
        if (part.empty()) {
            continue;
        }

        const auto equals = part.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument("caps annotation entries must use key=value");
        }

        const auto key = trim_to_string(part.substr(0, equals));
        const auto value = parse_string_token(part.substr(equals + 1));
        if (key.empty()) {
            throw std::invalid_argument("caps annotation key cannot be empty");
        }

        if (key == "caps") {
            if (caps_type) {
                throw std::invalid_argument("duplicate caps type annotation");
            }
            caps_type = value;
            continue;
        }

        if (params.contains(key)) {
            throw std::invalid_argument("duplicate caps parameter annotation");
        }
        params.emplace(key, value);
    }

    if (!caps_type || caps_type->empty()) {
        throw std::invalid_argument("pad caps annotation requires caps=TYPE");
    }

    return Caps(*caps_type, std::move(params));
}

std::map<std::string, std::string> parse_metadata_annotation(std::string_view text) {
    const auto metadata = parse_string_map_annotation(text, "metadata");
    if (metadata.empty()) {
        throw std::invalid_argument("metadata annotation requires at least one key=value entry");
    }

    return metadata;
}

ElementFactoryRegistry builtin_element_factory_registry(std::shared_ptr<leakflow::plot::PlotRuntime> plot_runtime) {
    if (!plot_runtime) {
        plot_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
    }
    return linked_element_factory_registry(std::move(plot_runtime));
}

Pipeline build_pipeline_from_expression(std::string_view expression, ElementFactoryRegistry element_factories) {
    std::ostringstream unused_output;
    ExpressionRunner runner(unused_output, std::move(element_factories),
                            std::make_shared<leakflow::plot::PlotRuntime>());
    return runner.build(expression);
}

PipelineExpressionBuildResult build_builtin_pipeline_from_expression(std::string_view expression) {
    auto plot_runtime = std::make_shared<leakflow::plot::PlotRuntime>();
    auto element_factories = builtin_element_factory_registry(plot_runtime);
    std::ostringstream unused_output;
    ExpressionRunner runner(unused_output, std::move(element_factories), plot_runtime);
    return {
        .pipeline = runner.build(expression),
        .plot_runtime = std::move(plot_runtime),
    };
}

int run_expression(std::string_view expression, std::ostream &output, RunExpressionOptions options) {
    ExpressionRunner runner(output);
    return runner.run(expression, options);
}

} // namespace leakflow::cli
