#pragma once

#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow::log {

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Off,
};

enum class LogColorMode {
    Auto,
    Never,
    Always,
};

struct LogRecord {
    LogLevel level = LogLevel::Info;
    std::string component;
    std::string message;
    std::string element;
    std::string element_name;
    std::string element_kclass;
    std::map<std::string, std::string> fields;
};

class LogFilter {
public:
    [[nodiscard]] static LogFilter parse(std::string_view text);

    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool matches(const LogRecord& record) const;
    [[nodiscard]] std::string to_string() const;

private:
    struct Clause {
        std::string key;
        std::string value;
    };

    std::vector<Clause> clauses_;
};

struct LogConfig {
    LogLevel level = LogLevel::Warning;
    LogColorMode color_mode = LogColorMode::Auto;
    LogFilter filter;
    bool summaries_enabled = true;
    std::int64_t summary_level = 2;
};

[[nodiscard]] LogLevel parse_log_level(std::string_view text);
[[nodiscard]] LogColorMode parse_log_color_mode(std::string_view text);
[[nodiscard]] bool parse_log_bool(std::string_view text);
[[nodiscard]] std::int64_t parse_summary_level(std::string_view text);

[[nodiscard]] std::string log_level_name(LogLevel level);
[[nodiscard]] std::string log_color_mode_name(LogColorMode mode);
[[nodiscard]] LogConfig default_config();
[[nodiscard]] LogConfig config_from_environment();

void configure(LogConfig config);
void configure(LogConfig config, std::ostream& output);
void reset_for_tests();

[[nodiscard]] const LogConfig& current_config();
[[nodiscard]] bool should_log(const LogRecord& record);
[[nodiscard]] bool summaries_enabled();
[[nodiscard]] std::int64_t summary_level();

void write(LogRecord record);

} // namespace leakflow::log
