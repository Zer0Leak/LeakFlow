#include "leakflow/core/linear_pipeline.hpp"
#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plugins/base/base_elements.hpp"
#include "leakflow/plugins/plot/plot_elements.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv)
{
    try {
        const std::string trace_path = argc > 1 ? argv[1] : LEAKFLOW_TUTORIAL_TRACE_PATH;

        auto plot_runtime = std::make_shared<leakflow::plot::PlotRuntime>();

        auto source = std::make_shared<leakflow::plugins::base::TorchSrc>("src");
        source->set_property("path", trace_path);

        auto trace_plot = std::make_shared<leakflow::plugins::plot::TracePlot>(plot_runtime, "plot");
        trace_plot->set_property("title", std::string("LeakFlow tutorial trace plot"));
        trace_plot->set_property("group", std::string("tutorial"));
        trace_plot->set_property("label", std::string("AES fixture"));
        trace_plot->set_property("layout", std::string("overlay"));

        leakflow::LinearPipeline pipeline;
        pipeline.add(source);
        pipeline.add(trace_plot);
        pipeline.link(source, "src", trace_plot, "sink");

        (void)pipeline.run();

        if (plot_runtime->has_sessions()) {
            leakflow::plot::run_until_closed(*plot_runtime);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "leakflow_trace_plot_tutorial: " << error.what() << '\n';
        return 1;
    }
}
