// Rezaeezade et al. (2025), "Breaking the Blindfold" -- PoI finder replication.
//
// This app aggregates the Pearson PoI finder across many capture folders. Each
// key_* folder holds one aligned (traces, plaintexts, key) capture. We feed one
// folder per streaming step into an AppSrc (application-fed live source); because
// AppSrc declares itself live, PearsonPoiFinder auto-selects its incremental mode
// and folds every folder into running correlation moments -- it never resets
// between folders, only at start(). The last emitted PoI is the aggregate over all
// folders.
//
// Pipeline (built once, fed by the app):
//
//   AppSrc@src
//     src_0 (traces) -> Tee@trace_tee -> @poi.features / @leakage.traces
//     src_1 (plain)  -> @leakage.plaintexts
//     src_2 (key)    -> @leakage.keys
//   AesLeakage@leakage -> @poi.targets
//   PearsonPoiFinder@poi  (auto -> incremental because upstream is live)
//
// Usage:
//   leakflow_rezaeezade_poi_finder [--graph] [ROOT_DIR]
//   ROOT_DIR defaults to traces/aes/sync/aes_sync_poi and must contain key_* dirs,
//   each with traces.pt, plain_texts.pt, and key.pt.

#include "leakflow_cli.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/plot/pipeline_graph.hpp"
#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plugins/base/app_src.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr auto default_root = "traces/aes/sync/aes_sync_poi";
constexpr auto traces_file = "traces.pt";
constexpr auto plaintexts_file = "plain_texts.pt";
constexpr auto key_file = "key.pt";

[[nodiscard]] torch::Tensor load_pt(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open " + path.string());
    }
    std::vector<char> data{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return torch::pickle_load(data).toTensor();
}

[[nodiscard]] leakflow::Buffer tensor_buffer(torch::Tensor tensor)
{
    auto payload = std::make_shared<leakflow::base::TorchTensorPayload>(std::move(tensor));
    leakflow::Buffer buffer(payload->caps());
    buffer.set_payload(payload);
    return buffer;
}

// Discover key_* capture folders under root, sorted for determinism.
[[nodiscard]] std::vector<fs::path> capture_folders(const fs::path& root)
{
    if (!fs::is_directory(root)) {
        throw std::runtime_error("root is not a directory: " + root.string());
    }

    std::vector<fs::path> folders;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory() && fs::exists(entry.path() / traces_file)) {
            folders.push_back(entry.path());
        }
    }
    std::sort(folders.begin(), folders.end());
    return folders;
}

// The analysis core is identical in both modes. With plotting on (--graph) a third
// Tee branch feeds traces to a TracePlot and the PoI is converted to annotation
// markers overlaid on it; headless ends at @poi so run() returns the PoI buffer for
// the text summary.
[[nodiscard]] std::string pipeline_expression(bool with_plot)
{
    std::string expression =
        "AppSrc@src; "
        "Tee@trace_tee; "
        "AesLeakage@leakage(channels=[HW(m),HW(y)],byte_indexes=[]); "
        "PearsonPoiFinder@poi(top_k=[50],rank_by=[abs]); ";

    if (with_plot) {
        expression +=
            "CorrelationPoiToPlotAnnotations@ann(precision=3); "
            "TracePlot@plot(title=\"AES aggregate PoIs\",group=aes,label=traces,"
            "update_mode=replace,x_axis=sample); ";
    }

    expression +=
        "@src.src_0 ! @trace_tee; "
        "@trace_tee.src_0 ! @poi.features; "
        "@trace_tee.src_1 ! @leakage.traces; "
        "@src.src_1 ! @leakage.plaintexts; "
        "@src.src_2 ! @leakage.keys; "
        "@leakage ! @poi.targets";

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
    std::cout << "Final aggregate PoI:\n"
              << "  correlation_mode  = " << result->metadata("payload.poi.correlation_mode") << "\n"
              << "  observation_count = " << result->metadata("payload.poi.observation_count") << "\n"
              << "  features_count    = " << result->metadata("payload.poi.features_count") << "\n"
              << "  payload           = " << (result->payload() ? result->payload()->type_name() : "none") << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        bool graph = false;
        bool auto_start = false;
        fs::path root = default_root;
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--graph") {
                graph = true;
            } else if (arg == "--auto-start") {
                auto_start = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "usage: leakflow_rezaeezade_poi_finder [--graph] [--auto-start] [ROOT_DIR]\n"
                          << "  --graph       open the live pipeline graph window (player controls),\n"
                          << "                just like 'leakflow run --graph'\n"
                          << "  --auto-start  with --graph, begin running on open (else opens Stopped)\n"
                          << "  ROOT_DIR      default: " << default_root << " (key_* dirs with "
                          << traces_file << ", " << plaintexts_file << ", " << key_file << ")\n";
                return 0;
            } else {
                root = arg;
            }
        }

        const auto folders = capture_folders(root);
        if (folders.empty()) {
            throw std::runtime_error("no capture folders with " + std::string(traces_file) + " under " + root.string());
        }
        std::cout << "Aggregating PoI over " << folders.size() << " capture folder(s) under " << root << "\n";

        auto built = leakflow::cli::build_builtin_pipeline_from_expression(pipeline_expression(graph));
        auto source_element = built.pipeline.element("src");
        auto* app_src = dynamic_cast<leakflow::plugins::base::AppSrc*>(source_element.get());
        if (app_src == nullptr) {
            throw std::runtime_error("expected element 'src' to be an AppSrc");
        }

        // Lazy pull: the source asks for folder `index` when the pump needs it, so
        // the graph window opens immediately and only ~one folder is in memory at a
        // time. Each returned frame is one aligned (traces, plaintexts, key) set; the
        // PoI accumulates across frames. AppSrc owns the index and resets it to 0 on
        // start(), so Stop -> Start re-streams. nullopt = end of stream.
        app_src->set_frame_producer(
            [folders](std::size_t index) -> std::optional<std::vector<leakflow::Buffer>> {
                if (index >= folders.size()) {
                    return std::nullopt;
                }
                const auto& folder = folders[index];
                auto traces = tensor_buffer(load_pt(folder / traces_file));
                traces.set_metadata("capture.source", "ChipWhisperer");
                traces.set_metadata("origin.folder", folder.filename().string());

                std::vector<leakflow::Buffer> frame;
                frame.reserve(3);
                frame.push_back(std::move(traces));
                frame.push_back(tensor_buffer(load_pt(folder / plaintexts_file)));
                frame.push_back(tensor_buffer(load_pt(folder / key_file)));
                std::cout << "  streaming " << folder.filename().string() << "\n";
                return frame;
            });

        if (graph) {
            // Open the live pipeline graph exactly like `leakflow run --graph`: drive
            // a PipelineSession (Start/Stop/Pause/Resume player controls) with the
            // graph window and telemetry on. Folders are pulled lazily as the pump
            // runs, so the window appears at once and streams folder-by-folder.
            leakflow::plot::PlotLoopOptions options;
            options.window_title = "LeakFlow Pipeline Graph";
            options.auto_start = auto_start;
            leakflow::PipelineSession session(std::move(built.pipeline));
            session.set_telemetry_enabled(true);
            // Terminal here is the TracePlot sink, not @poi, so the returned buffer is
            // not the PoI -- the result is shown live in the trace/annotation plot.
            (void)leakflow::plot::run_pipeline_graph_until_closed(session, *built.plot_runtime, options);
            std::cout << "graph closed\n";
            return 0;
        }

        print_poi_summary(built.pipeline.run());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "leakflow_rezaeezade_poi_finder: " << error.what() << "\n";
        return 1;
    }
}
