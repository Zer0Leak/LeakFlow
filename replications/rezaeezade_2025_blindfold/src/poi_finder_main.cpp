// Rezaeezade et al. (2025), "Breaking the Blindfold" -- PoI finder replication.
//
// This app aggregates the Pearson PoI finder across many capture files. Each
// key_NN.h5 file holds one aligned (traces, plaintexts, key) capture. We feed one
// file per streaming step into an AppSrc (application-fed live source); because
// AppSrc declares itself live, PearsonCorrelator auto-selects its incremental mode
// and folds every folder into running correlation moments -- it never resets
// between folders, only at start(). PoiSelect (stateless) then picks the top-k PoIs.
// The last emitted PoI is the aggregate over all folders.
//
// Pipeline (built once, fed by the app):
//
//   AppSrc@src
//     src_0 (traces) -> Tee@trace_tee -> @corr.features / @leakage.traces
//     src_1 (plain)  -> @leakage.plaintexts
//     src_2 (key)    -> @leakage.keys
//   AesLeakage@leakage -> @corr.targets
//   PearsonCorrelator@corr -> PoiSelect@poi  (auto -> incremental because upstream is live)
//
// Usage:
//   leakflow_rezaeezade_poi_finder [--graph] [ROOT_DIR]
//   ROOT_DIR defaults to traces/aes/sync/aes_sync_poi and must contain key_*.h5
//   files, each with /traces, /plaintexts, and /keys datasets.

#include "leakflow_cli.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/extras/hdf5_tensor_dataset_reader.hpp"
#include "leakflow/log/logger.hpp"
#include "leakflow/plot/pipeline_graph.hpp"
#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plugins/base/app_src.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr auto default_root = "traces/aes/sync/aes_sync_poi";
constexpr auto traces_dataset = "/traces";
constexpr auto plaintexts_dataset = "/plaintexts";
constexpr auto keys_dataset = "/keys";

void log_info(std::string message, std::map<std::string, std::string> fields = {})
{
    leakflow::log::write(leakflow::log::LogRecord{
        .level = leakflow::log::LogLevel::Info,
        .component = "replication",
        .message = std::move(message),
        .fields = std::move(fields),
    });
}

[[nodiscard]] leakflow::Buffer tensor_buffer(torch::Tensor tensor)
{
    auto payload = std::make_shared<leakflow::base::TorchTensorPayload>(std::move(tensor));
    leakflow::Buffer buffer(payload->caps());
    buffer.set_payload(payload);
    return buffer;
}

// Discover key_*.h5 capture files under root, sorted for determinism.
[[nodiscard]] std::vector<fs::path> capture_files(const fs::path& root)
{
    if (!fs::is_directory(root)) {
        throw std::runtime_error("root is not a directory: " + root.string());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".h5") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// The analysis core is identical in every mode. With plotting on (--graph) a third
// Tee branch feeds traces to a TracePlot and the PoI is converted to annotation
// markers overlaid on it. With --save-correlation, a Tee forks @corr to a
// BufferFileSink so each fold's (and thus the final aggregate) correlation is
// persisted. PoiSelect@poi is declared last so it stays the terminal whose PoI
// run() returns for the headless summary.
[[nodiscard]] std::string pipeline_expression(bool with_plot, bool with_save)
{
    std::string expression =
        "AppSrc@src; "
        "Tee@trace_tee; "
        "AesLeakage@leakage(channels=[HW(m),HW(y)],byte_indexes=[]); "  // all 16 bytes, HW(m) & HW(y)
        "PearsonCorrelator@corr; ";  // stateful: accumulates the correlation

    if (with_save) {
        expression +=
            "Tee@corr_tee; "                 // fork the correlation to the sink + selector
            "BufferFileSink@save; ";  // persists the aggregate correlation (path set after build)
    }

    expression += "PoiSelect@poi(top_k=[50],rank_by=[abs]); ";  // stateless: selects top-k PoIs

    if (with_plot) {
        expression +=
            "CorrelationPoiToPlotAnnotations@ann(precision=3); "
            "TracePlot@plot(title=\"AES aggregate PoIs\",group=aes,label=traces,"
            "update_mode=accumulate,x_axis=sample); ";
    }

    expression +=
        "@src.src_0 ! @trace_tee; "
        "@trace_tee.src_0 ! @corr.features; "
        "@trace_tee.src_1 ! @leakage.traces; "
        "@src.src_1 ! @leakage.plaintexts; "
        "@src.src_2 ! @leakage.keys; "
        "@leakage ! @corr.targets; ";

    if (with_save) {
        expression += "@corr ! @corr_tee; @corr_tee.src_0 ! @poi; @corr_tee.src_1 ! @save";
    } else {
        expression += "@corr ! @poi";
    }

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
        // This app reports its progress through the LeakFlow log at info level.
        // LEAKFLOW_LOG_LEVEL (and the other LEAKFLOW_LOG_* vars) still override it.
        auto log_config = leakflow::log::config_from_environment();
        if (std::getenv("LEAKFLOW_LOG_LEVEL") == nullptr) {
            log_config.level = leakflow::log::LogLevel::Info;
        }
        leakflow::log::configure(std::move(log_config));

        bool graph = false;
        bool auto_start = false;
        std::optional<std::string> save_correlation_path;
        fs::path root = default_root;
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--graph") {
                graph = true;
            } else if (arg == "--auto-start") {
                auto_start = true;
            } else if (arg == "--save-correlation") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("--save-correlation needs a PATH argument");
                }
                save_correlation_path = argv[++index];
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "usage: leakflow_rezaeezade_poi_finder [--graph] [--auto-start]\n"
                          << "                                      [--save-correlation PATH] [ROOT_DIR]\n"
                          << "  --graph              open the live pipeline graph window (player controls),\n"
                          << "                       just like 'leakflow run --graph'\n"
                          << "  --auto-start         with --graph, begin running on open (else Stopped)\n"
                          << "  --save-correlation P  persist the aggregate correlation to the HDF5 file P\n"
                          << "                       (reload later with BufferFileSrc to re-select PoIs offline)\n"
                          << "  ROOT_DIR             default: " << default_root
                          << " (key_*.h5 files with /traces, /plaintexts, /keys)\n"
                          << "  In --graph, the 'src' node exposes 'path' and 'max_trace_bundles'\n"
                          << "  controls (this app's AppSrc instance properties; editable when stopped).\n";
                return 0;
            } else {
                root = arg;
            }
        }

        // Validate the initial capture root up front for a clean startup error.
        const auto initial_bundles = capture_files(root);
        if (initial_bundles.empty()) {
            throw std::runtime_error("no key_*.h5 capture files under " + root.string());
        }
        log_info("aggregating PoI over trace bundles",
                 {{"bundles", std::to_string(initial_bundles.size())}, {"path", root.string()}});

        auto built = leakflow::cli::build_builtin_pipeline_from_expression(
            pipeline_expression(graph, save_correlation_path.has_value()));
        auto source_element = built.pipeline.element("src");
        auto* app_src = dynamic_cast<leakflow::plugins::base::AppSrc*>(source_element.get());
        if (app_src == nullptr) {
            throw std::runtime_error("expected element 'src' to be an AppSrc");
        }

        // App-exposed instance properties: the application enriches its own AppSrc
        // instance through the generic Element::add_property seam, so both render as
        // controls on the 'src' node in the --graph panel (the panel draws the element's
        // live property specs). Both are Lifecycle-effect, so the panel makes them
        // editable only in the Stopped state: configure the capture `path` and the
        // `max_trace_bundles` fold limit while stopped, then Start. The producer reads
        // them at stream start -- re-discovering `path` and capping by
        // `max_trace_bundles` (0 = all).
        const leakflow::PropertyEffect stopped_only_effect{
            .kind = leakflow::PropertyEffectKind::Lifecycle,
            .scope = leakflow::PropertyInvalidationScope::FullPipeline,
        };
        app_src->add_property(leakflow::PropertySpec(
            "path", root.string(), "capture root directory (one key_*.h5 per trace bundle)", "",
            std::monostate{}, "a directory of key_*.h5 captures", stopped_only_effect));
        app_src->add_property(leakflow::PropertySpec(
            "max_trace_bundles", std::int64_t{0}, "fold at most N trace bundles (0 = all)", "bundles",
            leakflow::IntRangeConstraint{0, std::numeric_limits<std::int64_t>::max()}, "",
            stopped_only_effect));

        if (save_correlation_path) {
            built.pipeline.element("save")->set_property("path", *save_correlation_path);
            log_info("saving aggregate correlation", {{"path", *save_correlation_path}});
        }

        // Lazy pull: one aligned (traces, plaintexts, key) trace bundle per frame; the
        // PoI accumulates across bundles. AppSrc rewinds to index 0 on start(), so a
        // Stop -> Start re-streams. The producer reads the app-exposed properties at
        // stream start: it (re)discovers `path` at index 0 (only when it changed) and
        // caps the fold count by `max_trace_bundles`. Both are edited while stopped, so
        // each run sees a fixed config. nullopt = end of stream.
        auto bundles = std::make_shared<std::vector<fs::path>>(initial_bundles);
        auto bundles_path = std::make_shared<std::string>(root.string());
        app_src->set_frame_producer(
            [app_src, bundles, bundles_path](std::size_t index, const auto& report)
                -> std::optional<std::vector<leakflow::Buffer>> {
                if (index == 0) {
                    const auto path = app_src->property_as<std::string>("path").value_or(*bundles_path);
                    if (path != *bundles_path) {
                        *bundles = capture_files(path);
                        *bundles_path = path;
                    }
                }
                const auto configured = app_src->property_as<std::int64_t>("max_trace_bundles").value_or(0);
                const auto count = (configured > 0)
                    ? std::min<std::size_t>(static_cast<std::size_t>(configured), bundles->size())
                    : bundles->size();
                if (index >= count) {
                    report(1.0, "done", count, count);
                    return std::nullopt;
                }
                const auto& file = (*bundles)[index];
                // Drive this AppSrc's --graph progress bar: the app is the only place
                // that knows the bundle count, so it reports; AppSrc relays it to the
                // generic element progress channel (per instance). See report(...).
                report(static_cast<double>(index) / static_cast<double>(count),
                       "streaming " + file.stem().string(), index, count);
                // One key_NN.h5 per fold: read its aligned /traces, /plaintexts, /keys
                // through the same HDF5 reader Hdf5FileSrc uses.
                leakflow::extras::Hdf5TensorDatasetReader reader(file);
                auto traces = tensor_buffer(reader.read_tensor(traces_dataset));
                traces.set_metadata("capture.source", "ChipWhisperer");
                traces.set_metadata("origin.file", file.filename().string());

                std::vector<leakflow::Buffer> frame;
                frame.reserve(3);
                frame.push_back(std::move(traces));
                frame.push_back(tensor_buffer(reader.read_tensor(plaintexts_dataset)));
                frame.push_back(tensor_buffer(reader.read_tensor(keys_dataset)));
                log_info("streaming trace bundle",
                         {{"file", file.filename().string()}, {"index", std::to_string(index)}});
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
            log_info("graph closed");
            return 0;
        }

        print_poi_summary(built.pipeline.run());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "leakflow_rezaeezade_poi_finder: " << error.what() << "\n";
        return 1;
    }
}
