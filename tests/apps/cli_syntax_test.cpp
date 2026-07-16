#include "leakflow_cli.hpp"
#include "leakflow/log/logger.hpp"
#include "leakflow/plugins/core/descriptor_catalog.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

template <typename Function>
std::optional<std::string> invalid_argument_message(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument& error) {
        return std::string(error.what());
    }

    return std::nullopt;
}

template <typename Function>
std::optional<std::string> exception_message(Function function)
{
    try {
        function();
    } catch (const std::exception& error) {
        return std::string(error.what());
    }

    return std::nullopt;
}

} // namespace

int main()
{
    if (!expect(leakflow::cli::normalized_identifier("Fake-Src") == "fakesrc",
            "normalized_identifier did not ignore punctuation and case")) {
        return 1;
    }

    leakflow::PropertySpec bool_spec("enabled", true);
    if (!expect(std::get<bool>(leakflow::cli::parse_property_value(bool_spec, "false")) == false,
            "bool false parsing failed")) {
        return 1;
    }
    if (!expect(std::get<bool>(leakflow::cli::parse_property_value(bool_spec, "1")) == true,
            "bool numeric parsing failed")) {
        return 1;
    }

    leakflow::PropertySpec int_spec(
        "poi_count",
        std::int64_t{20},
        "",
        "",
        leakflow::IntRangeConstraint{1, 100});
    if (!expect(std::get<std::int64_t>(leakflow::cli::parse_property_value(int_spec, "50")) == 50,
            "integer parsing failed")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([&int_spec] {
            (void)leakflow::cli::parse_property_value(int_spec, "0");
        }),
            "integer range validation was not applied")) {
        return 1;
    }

    leakflow::PropertySpec double_spec("threshold", 0.25);
    if (!expect(std::get<double>(leakflow::cli::parse_property_value(double_spec, "0.5")) == 0.5,
            "double parsing failed")) {
        return 1;
    }

    leakflow::PropertySpec string_spec("label", std::string("default"));
    if (!expect(std::get<std::string>(leakflow::cli::parse_property_value(string_spec, "pearson")) == "pearson",
            "bare string parsing failed")) {
        return 1;
    }
    if (!expect(std::get<std::string>(
                    leakflow::cli::parse_property_value(string_spec, "\"hello \\\"leakflow\\\"\""))
                == "hello \"leakflow\"",
            "quoted string parsing failed")) {
        return 1;
    }

    leakflow::PropertySpec int_interval_spec("sample_window", leakflow::IntInterval{0, 10});
    if (!expect(std::get<leakflow::IntInterval>(
                    leakflow::cli::parse_property_value(int_interval_spec, "1000..2500"))
                == leakflow::IntInterval{1000, 2500},
            "integer interval parsing failed")) {
        return 1;
    }

    leakflow::PropertySpec double_interval_spec("time_window", leakflow::DoubleInterval{0.0, 1.0});
    if (!expect(std::get<leakflow::DoubleInterval>(
                    leakflow::cli::parse_property_value(double_interval_spec, "0.25..2.5"))
                == leakflow::DoubleInterval{0.25, 2.5},
            "double interval parsing failed")) {
        return 1;
    }

    leakflow::PropertySpec int_list_spec("selected_bytes", leakflow::IntList{});
    if (!expect(std::get<leakflow::IntList>(
                    leakflow::cli::parse_property_value(int_list_spec, "[0,1,2,15]"))
                == leakflow::IntList{0, 1, 2, 15},
            "integer list parsing failed")) {
        return 1;
    }

    leakflow::PropertySpec double_list_spec("thresholds", leakflow::DoubleList{});
    if (!expect(std::get<leakflow::DoubleList>(
                    leakflow::cli::parse_property_value(double_list_spec, "[0.1,0.25,0.5]"))
                == leakflow::DoubleList{0.1, 0.25, 0.5},
            "double list parsing failed")) {
        return 1;
    }

    leakflow::PropertySpec string_list_spec("columns", leakflow::StringList{});
    if (!expect(std::get<leakflow::StringList>(
                    leakflow::cli::parse_property_value(string_list_spec, "[traces,\"plain texts\",key]"))
                == leakflow::StringList{"traces", "plain texts", "key"},
            "string list parsing failed")) {
        return 1;
    }
    if (!expect(std::get<leakflow::StringList>(
                    leakflow::cli::parse_property_value(string_list_spec, "[]"))
                == leakflow::StringList{},
            "empty string list parsing failed")) {
        return 1;
    }

    const auto caps = leakflow::cli::parse_caps_annotation("caps=sca/traces; dtype=float32; layout=trace,sample");
    if (!expect(caps.type() == "sca/traces", "caps type parsing failed")) {
        return 1;
    }
    if (!expect(caps.param("dtype") == "float32", "caps dtype parameter parsing failed")) {
        return 1;
    }
    if (!expect(caps.param("layout") == "trace,sample", "caps layout parameter parsing failed")) {
        return 1;
    }
    if (!expect(caps.to_string() == "sca/traces; dtype=float32; layout=trace,sample",
            "caps deterministic string form changed")) {
        return 1;
    }

    const auto quoted_caps = leakflow::cli::parse_caps_annotation("caps=\"tensor/torch\"; label=\"trace data\"");
    if (!expect(quoted_caps.type() == "tensor/torch", "quoted caps type parsing failed")) {
        return 1;
    }
    if (!expect(quoted_caps.param("label") == "trace data", "quoted caps parameter parsing failed")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            (void)leakflow::cli::parse_caps_annotation("dtype=float32");
        }),
            "missing caps type annotation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            (void)leakflow::cli::parse_caps_annotation("caps=sca/test; dtype=float32; dtype=float64");
        }),
            "duplicate caps parameter was not rejected")) {
        return 1;
    }

    const auto metadata = leakflow::cli::parse_metadata_annotation(
        "dataset=aes_sync; key_id=01; note=\"hello leakflow\"");
    if (!expect(metadata.at("dataset") == "aes_sync", "metadata dataset parsing failed")) {
        return 1;
    }
    if (!expect(metadata.at("key_id") == "01", "metadata key_id parsing failed")) {
        return 1;
    }
    if (!expect(metadata.at("note") == "hello leakflow", "quoted metadata parsing failed")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            (void)leakflow::cli::parse_metadata_annotation("");
        }),
            "empty metadata annotation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            (void)leakflow::cli::parse_metadata_annotation("dataset=a; dataset=b");
        }),
            "duplicate metadata key was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            (void)leakflow::cli::parse_metadata_annotation("dataset");
        }),
            "malformed metadata entry was not rejected")) {
        return 1;
    }

    std::ostringstream simple_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc ! Summary", simple_output) == 0,
            "simple run expression returned nonzero")) {
        return 1;
    }
    if (!expect(simple_output.str().find("caps=sca/fake") != std::string::npos,
            "simple run expression did not produce summary output")) {
        return 1;
    }

    std::ostringstream implicit_summary_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc", implicit_summary_output) == 0,
            "source-only run expression returned nonzero")) {
        return 1;
    }
    if (!expect(implicit_summary_output.str().empty(),
            "CLI printed an implicit terminal buffer summary")) {
        return 1;
    }

    leakflow::log::LogConfig flow_log_config;
    flow_log_config.level = leakflow::log::LogLevel::Info;
    flow_log_config.color_mode = leakflow::log::LogColorMode::Never;
    std::ostringstream flow_logs;
    leakflow::log::configure(flow_log_config, flow_logs);
    std::ostringstream flow_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc@src; FakeSink@sink; @src ! @sink", flow_output) == 0,
            "flow log run expression returned nonzero")) {
        return 1;
    }
    const auto flow_text = flow_logs.str();
    if (!expect(flow_text.find("pipeline: buffer flow") != std::string::npos,
            "INFO logs did not include a buffer flow record")) {
        return 1;
    }
    if (!expect(flow_text.find("FakeSrc@src.src -> FakeSink@sink.sink") != std::string::npos,
            "buffer flow record did not include source and sink endpoints")) {
        return 1;
    }
    if (!expect(flow_text.find("metadata (2):") != std::string::npos
                && flow_text.find("source=fake") != std::string::npos,
            "buffer flow record did not include metadata")) {
        return 1;
    }
    if (!expect(flow_text.find("payload:\n    type=none") != std::string::npos,
            "buffer flow record did not include payload facts")) {
        return 1;
    }
    if (!expect(flow_text.find("starting pipeline") == std::string::npos,
            "INFO logs still included lifecycle noise")) {
        return 1;
    }
    if (!expect(flow_text.find("buffer flow") == flow_text.rfind("buffer flow"),
            "buffer flow record was printed more than once for one route")) {
        return 1;
    }
    leakflow::log::reset_for_tests();

    leakflow::log::LogConfig summary_disabled_config;
    summary_disabled_config.summaries_enabled = false;
    std::ostringstream summary_disabled_logs;
    leakflow::log::configure(summary_disabled_config, summary_disabled_logs);

    std::ostringstream suppressed_summary_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc ! Summary", suppressed_summary_output) == 0,
            "summary-suppressed run expression returned nonzero")) {
        return 1;
    }
    if (!expect(suppressed_summary_output.str().empty(),
            "Summary output was not suppressed by the global summary setting")) {
        return 1;
    }

    std::ostringstream forced_summary_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc ! Summary(always_print=true)", forced_summary_output) == 0,
            "forced summary run expression returned nonzero")) {
        return 1;
    }
    if (!expect(forced_summary_output.str().find("caps=sca/fake") != std::string::npos,
            "Summary(always_print=true) did not override global summary suppression")) {
        return 1;
    }

    leakflow::log::reset_for_tests();

    std::ostringstream element_caps_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc[caps=sca/fake] ! Summary", element_caps_output) == 0,
            "element-level caps annotation failed")) {
        return 1;
    }
    if (!expect(element_caps_output.str().find("caps=sca/fake") != std::string::npos,
            "element-level caps annotation changed pipeline output")) {
        return 1;
    }

    std::ostringstream element_metadata_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc{dataset=smoke; note=\"hello\"} ! Summary(level=3)",
                    element_metadata_output)
                == 0,
            "element-level metadata annotation failed")) {
        return 1;
    }
    if (!expect(element_metadata_output.str().find("dataset=smoke") != std::string::npos
                && element_metadata_output.str().find("note=hello") != std::string::npos,
            "element-level metadata annotation was not stamped on output")) {
        return 1;
    }

    std::ostringstream multiline_annotation_output;
    if (!expect(leakflow::cli::run_expression(
                    "FakeSrc@src \\ \n"
                    "    (caps_type=sca/test) \\\n"
                    "    {dataset=smoke}; \\\n"
                    "@src ! Summary(level=3)",
                    multiline_annotation_output)
                == 0,
            "multiline expression with line-continuation backslashes failed")) {
        return 1;
    }
    if (!expect(multiline_annotation_output.str().find("caps=sca/test") != std::string::npos
                && multiline_annotation_output.str().find("dataset=smoke") != std::string::npos,
            "multiline expression did not preserve properties and metadata")) {
        return 1;
    }

    std::ostringstream combined_output;
    if (!expect(leakflow::cli::run_expression(
                    "FakeSrc(caps_type=sca/test)[caps=sca/fake; dtype=float32]{dataset=smoke; note=\"combined\"} ! Summary(level=3)",
                    combined_output)
                == 0,
            "combined property/caps/metadata expression failed")) {
        return 1;
    }
    if (!expect(combined_output.str().find("caps=sca/test") != std::string::npos,
            "combined property/caps/metadata expression did not set element property")) {
        return 1;
    }
    if (!expect(combined_output.str().find("dataset=smoke") != std::string::npos
                && combined_output.str().find("note=combined") != std::string::npos,
            "combined property/caps/metadata expression did not stamp metadata")) {
        return 1;
    }

    std::ostringstream tee_output;
    if (!expect(leakflow::cli::run_expression(
                    "FakeSrc ! Tee@t{dataset=smoke; branch=all}; @t.src_0{branch=summary} ! Summary(level=3); @t.src_1{branch=sink} ! Summary(level=3)",
                    tee_output)
                == 0,
            "tee run expression returned nonzero")) {
        return 1;
    }
    if (!expect(tee_output.str().find("dataset=smoke") != std::string::npos,
            "tee element-level metadata annotation was not stamped on output branches")) {
        return 1;
    }
    if (!expect(tee_output.str().find("branch=summary") != std::string::npos
                && tee_output.str().find("branch=sink") != std::string::npos,
            "tee exact output metadata annotations did not override all-output metadata")) {
        return 1;
    }

    std::ostringstream tee_template_output;
    if (!expect(leakflow::cli::run_expression(
                    "FakeSrc ! Tee@t; @t.src_%u{branch=template}; @t.src_0{branch=summary} ! Summary(level=3); @t.src_1 ! Summary(level=3)",
                    tee_template_output)
                == 0,
            "tee template metadata run expression returned nonzero")) {
        return 1;
    }
    if (!expect(tee_template_output.str().find("branch=summary") != std::string::npos
                && tee_template_output.str().find("branch=template") != std::string::npos,
            "tee template metadata annotation did not apply with exact-pad precedence")) {
        return 1;
    }

    std::ostringstream two_summary_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc ! Summary ! Summary ! FakeSink", two_summary_output) == 0,
            "two-summary run expression returned nonzero")) {
        return 1;
    }
    if (!expect(two_summary_output.str().find("Buffer@summary0") != std::string::npos,
            "first summary output did not include generated summary name")) {
        return 1;
    }
    if (!expect(two_summary_output.str().find("Buffer@summary1") != std::string::npos,
            "second summary output did not include generated summary name")) {
        return 1;
    }

    std::ostringstream name_property_output;
    if (!expect(leakflow::cli::run_expression(
                    "FakeSrc(name=src); Summary(name=summary,level=3); @src ! @summary",
                    name_property_output)
                == 0,
            "name property did not create a referenceable element name")) {
        return 1;
    }
    if (!expect(name_property_output.str().find("Buffer@summary") != std::string::npos,
            "Summary(name=summary) did not update the element instance name")) {
        return 1;
    }

    std::ostringstream generated_reference_output;
    if (!expect(leakflow::cli::run_expression("FakeSrc; Summary; @fakesrc0 ! @summary0",
                    generated_reference_output)
                == 0,
            "generated element names were not referenceable")) {
        return 1;
    }

    auto built = leakflow::cli::build_builtin_pipeline_from_expression(
        "FakeSrc ! Tee; @tee0.src_0 ! Summary; @tee0.src_1 ! FakeSink");
    if (!expect(built.pipeline.find_element("tee0") != nullptr,
            "built pipeline did not expose element lookup by name")) {
        return 1;
    }
    if (!expect(built.pipeline.elements_by_type("Tee").size() == 1,
            "built pipeline did not expose element lookup by type")) {
        return 1;
    }
    if (!expect(built.plot_runtime != nullptr, "built-in expression result did not retain PlotRuntime")) {
        return 1;
    }

    // The ML-to-plot bridge is registered by the built-in CLI factory with the
    // shared PlotRuntime. Build only (do not run) so this remains a headless
    // factory/linkage test and never opens a GUI window.
    {
        auto clustering_table = leakflow::cli::build_builtin_pipeline_from_expression(
            "ClusteringEvaluate@evaluation; "
            "ClusteringMetricsTablePlot@metrics(update_mode=accumulate); "
            "@evaluation.evaluation{payload.parameter.dataset=fixture} ! @metrics.sink");
        if (!expect(clustering_table.pipeline.elements_by_type("ClusteringMetricsTablePlot").size() == 1,
                "built-in factory did not create ClusteringMetricsTablePlot")) {
            return 1;
        }
        if (!expect(clustering_table.plot_runtime != nullptr && clustering_table.plot_runtime->views().size() > 1,
                "ML plot factory did not retain a registered shared PlotView")) {
            return 1;
        }
        const auto& links = clustering_table.pipeline.links();
        const auto linked = std::ranges::any_of(links, [](const auto& link) {
            return link.source_element->name() == "evaluation" && link.source_pad_name == "evaluation"
                && link.sink_element->name() == "metrics" && link.sink_pad_name == "sink";
        });
        if (!expect(linked, "ClusteringEvaluate was not wired to the table bridge")) {
            return 1;
        }
    }

    // Creation-with-pad: `Sync@s.in_0` creates and addresses an (on-request) input
    // pad inline, so a fan-in join can be wired without a separate declaration. The
    // join is declared after its inputs (source-before-sink add order).
    {
        auto join = leakflow::cli::build_builtin_pipeline_from_expression(
            "FakeSrc@a ! Queue@qa ; FakeSrc@b ! Queue@qb ; "
            "@qa.src ! Sync@s.in_0 ; @qb.src ! @s.in_1 ; @s.out_0 ! FakeSink ; @s.out_1 ! FakeSink");
        const auto &links = join.pipeline.links();
        const auto links_into = [&](const char *sink, const char *pad) {
            for (const auto &link : links) {
                if (link.sink_element->name() == sink && link.sink_pad_name == pad) {
                    return true;
                }
            }
            return false;
        };
        const auto links_from = [&](const char *source, const char *pad) {
            for (const auto &link : links) {
                if (link.source_element->name() == source && link.source_pad_name == pad) {
                    return true;
                }
            }
            return false;
        };
        if (!expect(join.pipeline.find_element("s") != nullptr, "creation-with-pad did not create the Sync element")) {
            return 1;
        }
        if (!expect(links_into("s", "in_0") && links_into("s", "in_1"),
                    "creation-with-pad did not wire Sync.in_0 / in_1")) {
            return 1;
        }
        if (!expect(links_from("s", "out_0") && links_from("s", "out_1"),
                    "creation-with-pad did not wire Sync.out_0 / out_1")) {
            return 1;
        }
    }

    // A pad selector that matches no template is rejected at parse time with the
    // exact element/pad context.
    const auto unknown_pad_message = invalid_argument_message([] {
            (void)leakflow::cli::build_builtin_pipeline_from_expression("FakeSrc@a ! Queue@qa ; @qa.src ! Sync@s.nope");
        });
    if (!expect(unknown_pad_message.has_value(),
            "creation-with-pad accepted an unknown pad")) {
        return 1;
    }
    if (!expect(
            *unknown_pad_message
                == "unknown pad '@s.nope' on element 'Sync@s'; input pads: (none); output pads: (none); "
                   "input pad templates: in_%u; output pad templates: out_%u",
            "creation-with-pad unknown-pad error did not identify the element and pad")) {
        return 1;
    }

    const auto reversed_declaration_message = invalid_argument_message([] {
        (void)leakflow::cli::build_builtin_pipeline_from_expression(
            "Queue@sink; Queue@source; @source.src ! @sink.sink");
    });
    if (!expect(
            reversed_declaration_message
                && *reversed_declaration_message
                    == "cannot link source endpoint '@source.src' to sink endpoint '@sink.sink': "
                       "source element '@source' was declared after sink element '@sink'; declare the "
                       "source before the sink",
            "CLI declaration-order error did not identify both link endpoints")) {
        return 1;
    }

    leakflow::ElementFactoryRegistry core_factories;
    leakflow::plugins::core::register_element_factories(core_factories);
    auto custom_pipeline = leakflow::cli::build_pipeline_from_expression(
        "FakeSrc ! Summary",
        std::move(core_factories));
    if (!expect(custom_pipeline.find_element("fakesrc0") != nullptr,
            "custom registry expression build did not create FakeSrc")) {
        return 1;
    }

    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc ! Tee@t; @t ! Summary", output);
        }),
            "ambiguous tee output was not rejected")) {
        return 1;
    }
    auto tee_torch_sink_message = exception_message([] {
        std::ostringstream output;
        (void)leakflow::cli::run_expression(
            "FakeSrc ! Tee@t; @t.src_1 ! TorchFileSink(path=/tmp/leakflow_unreachable.pt)",
            output);
    });
    if (!expect(tee_torch_sink_message
            && tee_torch_sink_message->find("TorchFileSink requires leakflow/torch-tensor input caps")
                != std::string::npos,
            "Tee ANY source pad should link, then fail at runtime for non-Torch buffers")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc ! Tee@t; @t[caps=sca/fake] ! Summary", output);
        }),
            "ambiguous element-level caps annotation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc ! Tee@t; @t.out_%u{branch=summary}", output);
        }),
            "unmatched pad-template metadata annotation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc ! Tee@t; @t.src_%u ! Summary", output);
        }),
            "pad-template link endpoint was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc@src; @src.missing ! Summary", output);
        }),
            "missing pad reference was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc@src; @src.src[dtype=float32]", output);
        }),
            "bad pad caps annotation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc@src; @src.src{dataset}", output);
        }),
            "bad pad metadata annotation was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc@x; Summary@x", output);
        }),
            "duplicate element name was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc(name=x); Summary(name=x)", output);
        }),
            "duplicate element name property was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc@src(name=other)", output);
        }),
            "conflicting @name and name property was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("FakeSrc@src; @src(name=other)", output);
        }),
            "post-creation element rename was not rejected")) {
        return 1;
    }
    if (!expect(throws_invalid_argument([] {
            std::ostringstream output;
            (void)leakflow::cli::run_expression("Summary(level=9)", output);
        }),
            "invalid property value was not rejected")) {
        return 1;
    }
    const auto element_token_message = invalid_argument_message([] {
        std::ostringstream output;
        (void)leakflow::cli::run_expression("FakeSrc@src \\ not_a_line_continuation", output);
    });
    if (!expect(element_token_message
                && element_token_message->find("unexpected token '\\\\' after element creation FakeSrc@src")
                    != std::string::npos,
            "element creation error did not include element name and unexpected token")) {
        return 1;
    }
    const auto reference_token_message = invalid_argument_message([] {
        std::ostringstream output;
        (void)leakflow::cli::run_expression("FakeSrc@src; @src.src$", output);
    });
    if (!expect(reference_token_message
                && reference_token_message->find("unexpected token '$' after element reference @src.src")
                    != std::string::npos,
            "element reference error did not include reference name and unexpected token")) {
        return 1;
    }

    return 0;
}
