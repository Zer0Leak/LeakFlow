#include "leakflow_cli.hpp"

#include "leakflow/plot/plot_runtime.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    try {
        const std::string traces_path = argc > 1 ? argv[1] : LEAKFLOW_TUTORIAL_TRACES_PATH;
        const std::string plaintexts_path = argc > 2 ? argv[2] : LEAKFLOW_TUTORIAL_PLAINTEXTS_PATH;
        const std::string key_path = argc > 3 ? argv[3] : LEAKFLOW_TUTORIAL_KEY_PATH;

        const auto expression =
            "TorchSrc@traces_src(path=" + traces_path
            + "){dataset=aes_sync_fixture; role=traces; sample_rate_hz=29454545.454545453}; "
              "TorchSrc@plain_src(path=" + plaintexts_path
            + "); "
              "TorchSrc@key_src(path=" + key_path
            + "); "
              "Tee@trace_tee; "
              "@trace_tee.src_%u{branch.family=trace-fanout}; "
              "@trace_tee.src_0{branch=plot-traces}; "
              "@trace_tee.src_1{branch=analysis-traces}; "
              "Tee@analysis_fanout; "
              "@analysis_fanout.src_%u{branch.family=analysis-fanout}; "
              "AesLeakage@leakage(byte_indexes=[0]); "
              "PearsonPoiFinder@poi(top_k=[10],rank_by=[abs]); "
              "CorrelationPoiToPlotAnnotations@ann(precision=3); "
              "TracePlot@plot(title=\"AES byte 0 PoIs\",group=aes,label=traces,x_axis=sample); "
              "@traces_src ! @trace_tee; "
              "@trace_tee.src_0 ! @plot.sink; "
              "@trace_tee.src_1 ! @analysis_fanout; "
              "@analysis_fanout.src_0 ! @leakage.traces; "
              "@analysis_fanout.src_1 ! @poi.features; "
              "@plain_src ! @leakage.plaintexts; "
              "@key_src ! @leakage.keys; "
              "@leakage ! @poi.targets; "
              "@poi ! @ann ! @plot.annotations";

        auto built = leakflow::cli::build_builtin_pipeline_from_expression(expression);

        auto plot = built.pipeline.element("plot");
        auto tees = built.pipeline.elements_by_type("Tee");
        std::cout << "Plot element: " << plot->element_type() << '@' << plot->name() << '\n';
        std::cout << "Tee elements: " << tees.size() << '\n';

        (void)built.pipeline.run();

        if (built.plot_runtime && built.plot_runtime->has_sessions()) {
            leakflow::plot::run_until_closed(*built.plot_runtime);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "leakflow_pipeline_expression_plot_tutorial: " << error.what() << '\n';
        return 1;
    }
}
