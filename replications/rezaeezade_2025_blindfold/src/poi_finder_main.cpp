// Rezaeezade et al. (2025), "Breaking the Blindfold" -- PoI finder replication.
//
// Reads one aligned HDF5 capture through Hdf5FileSrc and computes AES leakage,
// Pearson correlation, and the top-ranked points of interest.

// Usage:
//   leakflow_rezaeezade_poi_finder [--graph] [--auto-start]
//       [--save-correlation PATH] [CAPTURE.h5]


#include "leakflow_cli.hpp"

#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/plot/pipeline_graph.hpp"
#include "leakflow/plot/plot_runtime.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

constexpr auto default_capture = "traces/aes/sync/aes_sync_poi/key_01.h5";

[[nodiscard]] std::string pipeline_expression(
    const std::filesystem::path& capture, bool with_plot, bool with_save)
{
    std::string expression =
        "Hdf5FileSrc@src(path=\"" + capture.string() + "\"); "
        "Tee@trace_tee; "
        "AesLeakage@leakage(channels=[HW(m),HW(y)],byte_indexes=[]); "
        "PearsonCorrelator@corr; ";

    if (with_save) {
        expression += "Tee@corr_tee; BufferFileSink@save; ";
    }

    expression += "PoiSelect@poi(top_k=[50],rank_by=[abs]); ";

    if (with_plot) {
        expression +=
            "CorrelationPoiToPlotAnnotations@ann(precision=3); "
            "TracePlot@plot(title=\"AES PoIs\",group=aes,label=traces,"
            "update_mode=replace,x_axis=sample); ";
    }

    expression +=
        "@src.traces ! @trace_tee; "
        "@trace_tee.src_0 ! @corr.features; "
        "@trace_tee.src_1 ! @leakage.traces; "
        "@src.plaintexts ! @leakage.plaintexts; "
        "@src.keys ! @leakage.keys; "
        "@leakage ! @corr.targets; ";

    expression += (with_save
        ? "@corr ! @corr_tee; @corr_tee.src_0 ! @poi; @corr_tee.src_1 ! @save"
        : "@corr ! @poi");

    if (with_plot) {
        expression +=
            "; @trace_tee.src_2 ! @plot.sink; "
            "@poi ! @ann; "
            "@ann ! @plot.annotations";
    }
    return expression;
}

void print_poi_summary(const std::optional<leakflow::Buffer>& result)
{
    if (!result) {
        std::cout << "pipeline produced no terminal PoI buffer\n";
        return;
    }
    std::cout << "PoI result:\n"
              << "  correlation_mode  = " << result->metadata("payload.poi.correlation_mode") << '\n'
              << "  observation_count = " << result->metadata("payload.poi.observation_count") << '\n'
              << "  features_count    = " << result->metadata("payload.poi.features_count") << '\n'
              << "  payload           = "
              << (result->payload() ? result->payload()->type_name() : "none") << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        bool graph = false;
        bool auto_start = false;
        std::optional<std::string> save_correlation_path;
        std::filesystem::path capture = default_capture;

        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--graph") {
                graph = true;
            } else if (arg == "--auto-start") {
                auto_start = true;
            } else if (arg == "--save-correlation") {
                if (++index >= argc) {
                    throw std::runtime_error("--save-correlation needs a PATH argument");
                }
                save_correlation_path = argv[index];
            } else if (arg == "--help" || arg == "-h") {
                std::cout
                    << "usage: leakflow_rezaeezade_poi_finder [--graph] [--auto-start]\n"
                    << "                                      [--save-correlation PATH] [CAPTURE.h5]\n"
                    << "  CAPTURE.h5 defaults to " << default_capture << '\n';
                return 0;
            } else {
                capture = arg;
            }
        }

        if (!std::filesystem::is_regular_file(capture)) {
            throw std::runtime_error("capture is not a file: " + capture.string());
        }

        auto built = leakflow::cli::build_builtin_pipeline_from_expression(
            pipeline_expression(capture, graph, save_correlation_path.has_value()));
        if (save_correlation_path) {
            built.pipeline.element("save")->set_property("path", *save_correlation_path);
        }

        if (graph) {
            leakflow::plot::PlotLoopOptions options;
            options.window_title = "LeakFlow Pipeline Graph";
            options.auto_start = auto_start;
            leakflow::PipelineSession session(std::move(built.pipeline));
            session.set_telemetry_enabled(true);
            (void)leakflow::plot::run_pipeline_graph_until_closed(
                session, *built.plot_runtime, options);
            return 0;
        }

        print_poi_summary(built.pipeline.run());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "leakflow_rezaeezade_poi_finder: " << error.what() << '\n';
        return 1;
    }
}
