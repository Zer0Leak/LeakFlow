#include "leakflow/core/version.hpp"
#include "leakflow/log/logger.hpp"
#include "leakflow_cli.hpp"

#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void print_help() {
    std::cout << "Usage: leakflow [OPTIONS] [--help]\n";
    std::cout << "       leakflow [OPTIONS] run EXPRESSION\n";
    std::cout << "       leakflow [OPTIONS] run --graph EXPRESSION\n";
    std::cout << "       leakflow [OPTIONS] --pipeline NAME [--set "
                 "ELEMENT.PROPERTY=VALUE]\n";
    std::cout << '\n';
    std::cout << "Logging and summary options:\n";
    std::cout << "  --log-level LEVEL       off, error, warning, info, debug, trace\n";
    std::cout << "  --log-color MODE        auto, always, never\n";
    std::cout << "  --log-filter FILTER     e.g. element=TorchFileSrc,element_name=src\n";
    std::cout << "  --summaries             enable Summary output\n";
    std::cout << "  --no-summaries          disable Summary output unless "
                 "always_print=true\n";
    std::cout << "  --summary-level N       payload detail (0-3) in --graph tooltips and\n";
    std::cout << "                          the info buffer-flow log (default 2)\n";
    std::cout << '\n';
    std::cout << "Environment variables:\n";
    std::cout << "  LEAKFLOW_LOG_LEVEL, LEAKFLOW_LOG_COLOR, LEAKFLOW_LOG_FILTER, "
                 "LEAKFLOW_SUMMARIES, LEAKFLOW_SUMMARY_LEVEL\n";
    std::cout << '\n';
    std::cout << "Pipeline syntax examples:\n";
    std::cout << "  leakflow run 'FakeSrc ! Summary'\n";
    std::cout << "  leakflow run --graph 'FakeSrc ! Tee@t; @t.src_0 ! Summary; "
                 "@t.src_1 ! FakeSink'\n";
    std::cout << "  leakflow --log-level debug --log-filter element=FakeSrc run "
                 "'FakeSrc ! FakeSink'\n";
    std::cout << "  leakflow --no-summaries run 'FakeSrc ! "
                 "Summary(always_print=true)'\n";
    std::cout << "  leakflow run 'FakeSrc(caps_type=sca/test) ! Summary(level=2)'\n";
    std::cout << "  leakflow run 'FakeSrc ! Tee@t; @t.src_0 ! Summary; @t.src_1 "
                 "! FakeSink'\n";
    std::cout << "  leakflow run "
                 "'TorchFileSrc(path=tests/fixtures/aes/sync/key_01/"
                 "traces_first_50.pt) ! TracePlot(title=\"AES "
                 "traces\")'\n";
    std::cout << '\n';
    std::cout << "Phase 12 compatibility presets:\n";
    std::cout << "  fake-src-summary\n";
    std::cout << "  fake-src-tee-summary-sink\n";
}

[[nodiscard]] std::string join_expression_arguments(std::span<char *const> arguments) {
    std::string expression;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        if (index != 0) {
            expression += ' ';
        }
        expression += arguments[index];
    }
    return expression;
}

[[nodiscard]] std::string preset_property_updates(const std::vector<std::string> &assignments) {
    std::string updates;
    for (const auto &assignment : assignments) {
        const auto dot = assignment.find('.');
        const auto equals = assignment.find('=');
        if (dot == std::string::npos || equals == std::string::npos || dot > equals) {
            throw std::invalid_argument("property assignment must use ELEMENT.PROPERTY=VALUE");
        }

        updates += "; @";
        updates += assignment.substr(0, dot);
        updates += '(';
        updates += assignment.substr(dot + 1, equals - dot - 1);
        updates += '=';
        updates += assignment.substr(equals + 1);
        updates += ')';
    }
    return updates;
}

[[nodiscard]] std::string preset_expression(std::string_view name, const std::vector<std::string> &assignments) {
    const auto updates = preset_property_updates(assignments);

    if (name == "fake-src-summary") {
        return "FakeSrc@fake_src; Summary@summary" + updates + "; @fake_src ! @summary";
    }
    if (name == "fake-src-tee-summary-sink") {
        return "FakeSrc@fake_src; Tee@tee; Summary@summary; FakeSink@fake_sink" + updates +
               "; @fake_src ! @tee; @tee.src_0 ! @summary; @tee.src_1 ! @fake_sink";
    }

    throw std::invalid_argument("unknown pipeline preset");
}

int run_preset(std::string_view pipeline_name, const std::vector<std::string> &assignments) {
    return leakflow::cli::run_expression(preset_expression(pipeline_name, assignments), std::cout);
}

} // namespace

int main(int argc, char **argv) {
    try {
        auto log_config = leakflow::log::config_from_environment();

        int command_index = 1;
        while (command_index < argc) {
            const std::string_view argument(argv[command_index]);

            if (argument == "--log-level") {
                if (command_index + 1 >= argc) {
                    std::cerr << "--log-level requires a value\n";
                    return 1;
                }
                log_config.level = leakflow::log::parse_log_level(argv[++command_index]);
                ++command_index;
                continue;
            }

            if (argument == "--log-color") {
                if (command_index + 1 >= argc) {
                    std::cerr << "--log-color requires a value\n";
                    return 1;
                }
                log_config.color_mode = leakflow::log::parse_log_color_mode(argv[++command_index]);
                ++command_index;
                continue;
            }

            if (argument == "--log-filter") {
                if (command_index + 1 >= argc) {
                    std::cerr << "--log-filter requires a value\n";
                    return 1;
                }
                log_config.filter = leakflow::log::LogFilter::parse(argv[++command_index]);
                ++command_index;
                continue;
            }

            if (argument == "--summaries") {
                log_config.summaries_enabled = true;
                ++command_index;
                continue;
            }

            if (argument == "--no-summaries") {
                log_config.summaries_enabled = false;
                ++command_index;
                continue;
            }

            if (argument == "--summary-level") {
                if (command_index + 1 >= argc) {
                    std::cerr << "--summary-level requires a value\n";
                    return 1;
                }
                log_config.summary_level = leakflow::log::parse_summary_level(argv[++command_index]);
                ++command_index;
                continue;
            }

            break;
        }

        leakflow::log::configure(std::move(log_config));

        if (command_index >= argc) {
            std::cout << leakflow::build_banner() << '\n';
            return 0;
        }

        const std::string_view first_argument(argv[command_index]);
        if (first_argument == "--help" || first_argument == "-h") {
            print_help();
            return 0;
        }

        if (first_argument == "run") {
            auto graph = false;
            auto expression_index = command_index + 1;
            while (expression_index < argc) {
                const std::string_view run_option(argv[expression_index]);
                if (run_option == "--graph") {
                    graph = true;
                    ++expression_index;
                    continue;
                }
                break;
            }

            if (expression_index >= argc) {
                std::cerr << "leakflow: run requires a pipeline expression\n";
                return 1;
            }

            return leakflow::cli::run_expression(
                join_expression_arguments(
                    std::span(argv + expression_index, static_cast<std::size_t>(argc - expression_index))),
                std::cout, leakflow::cli::RunExpressionOptions{.graph = graph});
        }

        std::string pipeline_name;
        std::vector<std::string> assignments;

        for (int index = command_index; index < argc; ++index) {
            const std::string_view argument(argv[index]);

            if (argument == "--pipeline") {
                if (index + 1 >= argc) {
                    std::cerr << "--pipeline requires a value\n";
                    return 1;
                }
                pipeline_name = argv[++index];
                continue;
            }

            if (argument == "--set") {
                if (index + 1 >= argc) {
                    std::cerr << "--set requires a value\n";
                    return 1;
                }
                assignments.emplace_back(argv[++index]);
                continue;
            }

            std::cerr << "unknown argument: " << argument << '\n';
            return 1;
        }

        if (!pipeline_name.empty()) {
            return run_preset(pipeline_name, assignments);
        }
    } catch (const std::exception &error) {
        std::cerr << "leakflow: " << error.what() << '\n';
        return 1;
    }

    std::cerr << "unknown or missing command\n";
    print_help();
    return 1;
}
