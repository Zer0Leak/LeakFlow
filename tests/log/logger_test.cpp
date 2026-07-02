#include "leakflow/log/logger.hpp"

#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

template <typename Function>
bool throws_invalid_argument(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }

    return false;
}

} // namespace

int main()
{
    namespace log = leakflow::log;

    if (!expect(log::parse_log_level("trace") == log::LogLevel::Trace, "trace level parsing failed")) {
        return 1;
    }
    if (!expect(log::parse_log_level("warn") == log::LogLevel::Warning, "warn alias parsing failed")) {
        return 1;
    }
    if (!expect(log::parse_log_color_mode("force") == log::LogColorMode::Always,
            "force color alias parsing failed")) {
        return 1;
    }
    if (!expect(log::parse_log_bool("enabled"), "enabled bool parsing failed")) {
        return 1;
    }
    if (!expect(!log::parse_log_bool("off"), "off bool parsing failed")) {
        return 1;
    }

    if (!expect(log::parse_summary_level("1") == 1, "summary level 1 parsing failed")) {
        return 1;
    }
    if (!expect(log::parse_summary_level(" 3 ") == 3, "summary level trim parsing failed")) {
        return 1;
    }
    if (!expect(log::default_config().summary_level == 2, "default summary level should be 2")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] { (void)log::parse_summary_level("9"); }),
            "out-of-range summary level should throw")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] { (void)log::parse_summary_level("oops"); }),
            "non-integer summary level should throw")) {
        return 1;
    }

    const auto filter = log::LogFilter::parse("element=NumpySrc, element_name=src, element_kclass=Source");
    if (!expect(!filter.empty(), "filter should not be empty")) {
        return 1;
    }
    if (!expect(filter.to_string() == "element=NumpySrc,element_name=src,element_kclass=Source",
            "filter string form changed")) {
        return 1;
    }

    log::LogRecord matching{
        .level = log::LogLevel::Info,
        .component = "pipeline",
        .message = "loaded",
        .element = "NumpySrc",
        .element_name = "src",
        .element_kclass = "Source",
    };
    if (!expect(filter.matches(matching), "filter did not match the expected record")) {
        return 1;
    }

    auto nonmatching = matching;
    nonmatching.element_name = "other";
    if (!expect(!filter.matches(nonmatching), "filter matched the wrong element_name")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            (void)log::LogFilter::parse("field.path=demo.npy");
        }),
            "unsupported filter key was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            (void)log::LogFilter::parse("element=NumpySrc,broken");
        }),
            "malformed filter clause was not rejected")) {
        return 1;
    }

    std::ostringstream output;
    log::LogConfig config;
    config.level = log::LogLevel::Debug;
    config.color_mode = log::LogColorMode::Never;
    config.filter = log::LogFilter::parse("element=NumpySrc,element_name=src");
    log::configure(config, output);

    log::write(nonmatching);
    log::write(matching);
    const auto text = output.str();
    if (!expect(text.find("[info] pipeline: loaded") != std::string::npos,
            "log output did not include the level, component, and message")) {
        return 1;
    }
    if (!expect(text.find("element=NumpySrc") != std::string::npos,
            "log output did not include the element field")) {
        return 1;
    }
    if (!expect(text.find("element_name=other") == std::string::npos,
            "filter did not suppress the nonmatching record")) {
        return 1;
    }

    log::LogConfig off_config;
    off_config.level = log::LogLevel::Off;
    std::ostringstream off_output;
    log::configure(off_config, off_output);
    log::write(matching);
    if (!expect(off_output.str().empty(), "off log level emitted output")) {
        return 1;
    }

    log::reset_for_tests();
    return 0;
}
