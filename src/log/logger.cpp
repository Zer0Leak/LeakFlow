#include "leakflow/log/logger.hpp"

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow::log {
namespace {

struct LogState {
    LogConfig config = default_config();
    std::shared_ptr<spdlog::logger> logger;
};

[[nodiscard]] LogState& state()
{
    static LogState value;
    return value;
}

[[nodiscard]] std::mutex& state_mutex()
{
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] std::string trim_to_string(std::string_view text)
{
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }

    return std::string(text);
}

[[nodiscard]] std::string lower_string(std::string_view text)
{
    std::string lowered;
    lowered.reserve(text.size());
    for (const auto character : text) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    return lowered;
}

[[nodiscard]] std::vector<std::string_view> split(std::string_view text, char delimiter)
{
    std::vector<std::string_view> parts;
    std::size_t begin = 0;

    for (std::size_t index = 0; index < text.size(); ++index) {
        if (text[index] == delimiter) {
            parts.push_back(text.substr(begin, index - begin));
            begin = index + 1;
        }
    }

    parts.push_back(text.substr(begin));
    return parts;
}

[[nodiscard]] bool supported_filter_key(std::string_view key)
{
    return key == "component" || key == "element" || key == "element_name" || key == "element_kclass";
}

[[nodiscard]] std::optional<std::string_view> record_value_for(const LogRecord& record, std::string_view key)
{
    if (key == "component") {
        return record.component;
    }
    if (key == "element") {
        return record.element;
    }
    if (key == "element_name") {
        return record.element_name;
    }
    if (key == "element_kclass") {
        return record.element_kclass;
    }

    return std::nullopt;
}

[[nodiscard]] spdlog::level::level_enum to_spdlog_level(LogLevel level)
{
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Off:
        return spdlog::level::off;
    }

    return spdlog::level::off;
}

[[nodiscard]] spdlog::color_mode to_spdlog_color_mode(LogColorMode mode)
{
    switch (mode) {
    case LogColorMode::Auto:
        return spdlog::color_mode::automatic;
    case LogColorMode::Never:
        return spdlog::color_mode::never;
    case LogColorMode::Always:
        return spdlog::color_mode::always;
    }

    return spdlog::color_mode::automatic;
}

[[nodiscard]] std::shared_ptr<spdlog::logger> make_stderr_logger(const LogConfig& config)
{
    auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>(to_spdlog_color_mode(config.color_mode));
    auto logger = std::make_shared<spdlog::logger>("leakflow", std::move(sink));
    logger->set_pattern("[%^%l%$] %v");
    logger->set_level(to_spdlog_level(config.level));
    logger->flush_on(spdlog::level::warn);
    return logger;
}

[[nodiscard]] std::shared_ptr<spdlog::logger> make_ostream_logger(const LogConfig& config, std::ostream& output)
{
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(output, true);
    auto logger = std::make_shared<spdlog::logger>("leakflow-test", std::move(sink));
    logger->set_pattern("[%l] %v");
    logger->set_level(to_spdlog_level(config.level));
    logger->flush_on(spdlog::level::warn);
    return logger;
}

void ensure_configured()
{
    auto& value = state();
    if (!value.logger) {
        value.logger = make_stderr_logger(value.config);
    }
}

[[nodiscard]] bool level_is_enabled(LogLevel record_level, LogLevel configured_level)
{
    if (configured_level == LogLevel::Off || record_level == LogLevel::Off) {
        return false;
    }

    return static_cast<int>(record_level) >= static_cast<int>(configured_level);
}

void append_field(std::ostringstream& output, std::string_view key, std::string_view value, bool& first)
{
    if (value.empty()) {
        return;
    }

    if (!first) {
        output << ' ';
    }
    first = false;
    output << key << '=' << value;
}

[[nodiscard]] std::string format_record(const LogRecord& record)
{
    std::ostringstream output;

    if (!record.component.empty()) {
        output << record.component << ": ";
    }
    output << record.message;

    if (record.message.find('\n') != std::string::npos) {
        return output.str();
    }

    bool first = true;
    std::ostringstream fields;
    append_field(fields, "element", record.element, first);
    append_field(fields, "element_name", record.element_name, first);
    append_field(fields, "element_kclass", record.element_kclass, first);
    for (const auto& [key, value] : record.fields) {
        append_field(fields, key, value, first);
    }

    const auto field_text = fields.str();
    if (!field_text.empty()) {
        output << " [" << field_text << ']';
    }

    return output.str();
}

void apply_environment_value(LogConfig& config, const char* name)
{
    const auto* value = std::getenv(name);
    if (value == nullptr) {
        return;
    }

    const std::string_view text(value);
    if (std::string_view(name) == "LEAKFLOW_LOG_LEVEL") {
        config.level = parse_log_level(text);
        return;
    }
    if (std::string_view(name) == "LEAKFLOW_LOG_COLOR") {
        config.color_mode = parse_log_color_mode(text);
        return;
    }
    if (std::string_view(name) == "LEAKFLOW_LOG_FILTER") {
        config.filter = LogFilter::parse(text);
        return;
    }
    if (std::string_view(name) == "LEAKFLOW_SUMMARIES") {
        config.summaries_enabled = parse_log_bool(text);
        return;
    }
    if (std::string_view(name) == "LEAKFLOW_SUMMARY_LEVEL") {
        config.summary_level = parse_summary_level(text);
        return;
    }
}

} // namespace

LogFilter LogFilter::parse(std::string_view text)
{
    LogFilter filter;
    const auto trimmed = trim_to_string(text);
    if (trimmed.empty()) {
        return filter;
    }

    for (const auto raw_part : split(trimmed, ',')) {
        const auto part = trim_to_string(raw_part);
        if (part.empty()) {
            continue;
        }

        const auto equals = part.find('=');
        if (equals == std::string::npos) {
            throw std::invalid_argument("log filter clauses must use key=value");
        }

        const auto key = trim_to_string(std::string_view(part).substr(0, equals));
        const auto value = trim_to_string(std::string_view(part).substr(equals + 1));
        if (key.empty()) {
            throw std::invalid_argument("log filter key cannot be empty");
        }
        if (value.empty()) {
            throw std::invalid_argument("log filter value cannot be empty");
        }
        if (!supported_filter_key(key)) {
            throw std::invalid_argument("unsupported log filter key");
        }

        filter.clauses_.push_back(Clause{key, value});
    }

    return filter;
}

bool LogFilter::empty() const
{
    return clauses_.empty();
}

bool LogFilter::matches(const LogRecord& record) const
{
    for (const auto& clause : clauses_) {
        const auto value = record_value_for(record, clause.key);
        if (!value || *value != clause.value) {
            return false;
        }
    }

    return true;
}

std::string LogFilter::to_string() const
{
    std::ostringstream output;
    for (std::size_t index = 0; index < clauses_.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << clauses_[index].key << '=' << clauses_[index].value;
    }
    return output.str();
}

LogLevel parse_log_level(std::string_view text)
{
    const auto level = lower_string(trim_to_string(text));
    if (level == "trace") {
        return LogLevel::Trace;
    }
    if (level == "debug") {
        return LogLevel::Debug;
    }
    if (level == "info") {
        return LogLevel::Info;
    }
    if (level == "warning" || level == "warn") {
        return LogLevel::Warning;
    }
    if (level == "error" || level == "err") {
        return LogLevel::Error;
    }
    if (level == "off" || level == "none" || level == "false" || level == "0") {
        return LogLevel::Off;
    }

    throw std::invalid_argument("unknown log level");
}

LogColorMode parse_log_color_mode(std::string_view text)
{
    const auto mode = lower_string(trim_to_string(text));
    if (mode == "auto") {
        return LogColorMode::Auto;
    }
    if (mode == "never" || mode == "none" || mode == "off" || mode == "false" || mode == "0") {
        return LogColorMode::Never;
    }
    if (mode == "always" || mode == "force" || mode == "on" || mode == "true" || mode == "1") {
        return LogColorMode::Always;
    }

    throw std::invalid_argument("unknown log color mode");
}

bool parse_log_bool(std::string_view text)
{
    const auto value = lower_string(trim_to_string(text));
    if (value == "1" || value == "true" || value == "on" || value == "yes" || value == "enabled") {
        return true;
    }
    if (value == "0" || value == "false" || value == "off" || value == "no" || value == "disabled") {
        return false;
    }

    throw std::invalid_argument("unknown boolean log setting");
}

std::int64_t parse_summary_level(std::string_view text)
{
    const auto trimmed = trim_to_string(text);
    std::int64_t value = 0;
    const auto* begin = trimmed.data();
    const auto* end = begin + trimmed.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc() || ptr != end) {
        throw std::invalid_argument("summary level must be an integer between 0 and 3");
    }
    if (value < 0 || value > 3) {
        throw std::invalid_argument("summary level must be between 0 and 3");
    }

    return value;
}

std::string log_level_name(LogLevel level)
{
    switch (level) {
    case LogLevel::Trace:
        return "trace";
    case LogLevel::Debug:
        return "debug";
    case LogLevel::Info:
        return "info";
    case LogLevel::Warning:
        return "warning";
    case LogLevel::Error:
        return "error";
    case LogLevel::Off:
        return "off";
    }

    return "off";
}

std::string log_color_mode_name(LogColorMode mode)
{
    switch (mode) {
    case LogColorMode::Auto:
        return "auto";
    case LogColorMode::Never:
        return "never";
    case LogColorMode::Always:
        return "always";
    }

    return "auto";
}

LogConfig default_config()
{
    return {};
}

LogConfig config_from_environment()
{
    auto config = default_config();
    apply_environment_value(config, "LEAKFLOW_LOG_LEVEL");
    apply_environment_value(config, "LEAKFLOW_LOG_COLOR");
    apply_environment_value(config, "LEAKFLOW_LOG_FILTER");
    apply_environment_value(config, "LEAKFLOW_SUMMARIES");
    apply_environment_value(config, "LEAKFLOW_SUMMARY_LEVEL");
    return config;
}

void configure(LogConfig config)
{
    std::lock_guard lock(state_mutex());
    auto& value = state();
    value.config = std::move(config);
    value.logger = make_stderr_logger(value.config);
}

void configure(LogConfig config, std::ostream& output)
{
    std::lock_guard lock(state_mutex());
    auto& value = state();
    value.config = std::move(config);
    value.logger = make_ostream_logger(value.config, output);
}

void reset_for_tests()
{
    configure(default_config());
}

const LogConfig& current_config()
{
    std::lock_guard lock(state_mutex());
    ensure_configured();
    return state().config;
}

bool should_log(const LogRecord& record)
{
    std::lock_guard lock(state_mutex());
    ensure_configured();
    const auto& config = state().config;
    return level_is_enabled(record.level, config.level)
        && (config.filter.empty() || config.filter.matches(record));
}

bool summaries_enabled()
{
    std::lock_guard lock(state_mutex());
    ensure_configured();
    return state().config.summaries_enabled;
}

std::int64_t summary_level()
{
    std::lock_guard lock(state_mutex());
    ensure_configured();
    return state().config.summary_level;
}

void write(LogRecord record)
{
    std::lock_guard lock(state_mutex());
    ensure_configured();

    const auto& config = state().config;
    if (!level_is_enabled(record.level, config.level)) {
        return;
    }
    if (!config.filter.empty() && !config.filter.matches(record)) {
        return;
    }

    state().logger->log(to_spdlog_level(record.level), "{}", format_record(record));
}

} // namespace leakflow::log
