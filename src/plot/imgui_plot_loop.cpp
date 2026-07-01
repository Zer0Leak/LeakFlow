#include "leakflow/plot/plot_runtime.hpp"

#include "leakflow/core/pipeline_session.hpp"
#include "leakflow/log/logger.hpp"
#include "leakflow/plot/pipeline_graph.hpp"
#include "leakflow/plot/plot_view.hpp"
#include "leakflow/plot/trace_view.hpp"
#include "plot_render_util.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <implot_internal.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plot {
namespace {

#if defined(__linux__)
[[nodiscard]] bool environment_variable_is_set(const char *name) {
    const auto *value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

void require_display_environment() {
    if (environment_variable_is_set("DISPLAY")) {
        return;
    }
    if (environment_variable_is_set("WAYLAND_DISPLAY") && environment_variable_is_set("XDG_RUNTIME_DIR")) {
        return;
    }

    throw std::runtime_error("could not initialize GLFW for LeakFlow plot runtime: no DISPLAY or "
                             "WAYLAND_DISPLAY/XDG_RUNTIME_DIR is set. In Docker, start the container "
                             "with X11 forwarding (-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix) or "
                             "Wayland socket forwarding.");
}

void configure_glfw_platform_hint() {
#if defined(GLFW_PLATFORM)
    if (environment_variable_is_set("DISPLAY")) {
#if defined(GLFW_PLATFORM_X11)
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
        return;
    }

    if (environment_variable_is_set("WAYLAND_DISPLAY")) {
#if defined(GLFW_PLATFORM_WAYLAND)
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
#endif
    }
#endif
}
#endif

struct GlfwContext {
    GlfwContext() {
        glfwSetErrorCallback([](int, const char *description) {
            log::LogRecord record{
                .level = log::LogLevel::Error,
                .component = "plot",
                .message = "GLFW error",
                .fields =
                    {
                        {"description", description == nullptr ? "" : description},
                    },
            };
            log::write(std::move(record));
        });

#if defined(__linux__)
        require_display_environment();
        configure_glfw_platform_hint();
#endif

        if (glfwInit() == 0) {
            throw std::runtime_error("could not initialize GLFW for LeakFlow plot runtime; check display "
                                     "forwarding, X11/Wayland authorization, and Docker NVIDIA graphics "
                                     "driver capabilities");
        }
    }

    ~GlfwContext() { glfwTerminate(); }
};

struct GlfwWindow {
    GLFWwindow *window = nullptr;

    GlfwWindow(int width, int height, const std::string &title) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            throw std::runtime_error("could not create GLFW window for LeakFlow plot runtime");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
    }

    ~GlfwWindow() {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }
};

struct ImGuiContextOwner {
    explicit ImGuiContextOwner(GLFWwindow *window) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");
    }

    ~ImGuiContextOwner() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
};


void append_u32_be(std::vector<unsigned char> &output, std::uint32_t value) {
    output.push_back(static_cast<unsigned char>((value >> 24U) & 0xFFU));
    output.push_back(static_cast<unsigned char>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<unsigned char>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<unsigned char>(value & 0xFFU));
}

void append_chunk(std::vector<unsigned char> &png, const char *type, const std::vector<unsigned char> &data) {
    append_u32_be(png, static_cast<std::uint32_t>(data.size()));

    const auto type_begin = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());

    const auto crc =
        crc32(0, reinterpret_cast<const Bytef *>(png.data() + type_begin), static_cast<uInt>(png.size() - type_begin));
    append_u32_be(png, static_cast<std::uint32_t>(crc));
}

void save_rgba_png(const std::filesystem::path &path, int width, int height,
                   const std::vector<unsigned char> &rgba_bottom_up) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("PNG dimensions must be positive");
    }

    const auto stride = static_cast<std::size_t>(width) * 4U;
    std::vector<unsigned char> scanlines((stride + 1U) * static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        const auto destination = static_cast<std::size_t>(row) * (stride + 1U);
        const auto source = static_cast<std::size_t>(height - row - 1) * stride;
        scanlines[destination] = 0;
        std::memcpy(scanlines.data() + destination + 1U, rgba_bottom_up.data() + source, stride);
    }

    auto compressed_size = compressBound(static_cast<uLong>(scanlines.size()));
    std::vector<unsigned char> compressed(compressed_size);
    const auto compressed_result = compress2(compressed.data(), &compressed_size, scanlines.data(),
                                             static_cast<uLong>(scanlines.size()), Z_BEST_SPEED);
    if (compressed_result != Z_OK) {
        throw std::runtime_error("could not compress PNG image data");
    }
    compressed.resize(compressed_size);

    std::vector<unsigned char> png{
        0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n',
    };

    std::vector<unsigned char> ihdr;
    append_u32_be(ihdr, static_cast<std::uint32_t>(width));
    append_u32_be(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8);
    ihdr.push_back(6);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);

    append_chunk(png, "IHDR", ihdr);
    append_chunk(png, "IDAT", compressed);
    append_chunk(png, "IEND", {});

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not open PNG output path");
    }
    output.write(reinterpret_cast<const char *>(png.data()), static_cast<std::streamsize>(png.size()));
    if (!output) {
        throw std::runtime_error("could not write PNG output path");
    }
}

void save_current_framebuffer_png(const std::filesystem::path &path, int width, int height) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    save_rgba_png(path, width, height, pixels);
}

} // namespace

void draw_plot_runtime(PlotRuntime &runtime, PipelineControlRuntime *control_runtime) {
    const auto lock = std::scoped_lock(runtime.mutex());
    if (!runtime.has_sessions()) {
        ImGui::Begin("LeakFlow TracePlot");
        ImGui::TextUnformatted("No plot sessions");
        ImGui::End();
        return;
    }

    // Accumulate sliders (TraceView) auto-follow the newest trace only while the
    // session is Running; in Idle/Paused (or a plain non-session plot run) they stay
    // put. The control runtime lets a view's gear button open its element's panel.
    const PlotDrawContext context{
        .streaming = control_runtime != nullptr && control_runtime->session() != nullptr &&
                     control_runtime->session()->state() == leakflow::PipelineSessionState::Running,
        .control_runtime = control_runtime,
    };

    // Every plot type is a registered PlotView (TraceView, ScoreView, ...); the loop
    // is type-agnostic. Adding a plot type is a new PlotView, not an edit here.
    for (const auto &view : runtime.views()) {
        if (view) {
            view->draw(context);
        }
    }
}

void run_until_closed_impl(PlotRuntime &runtime, PipelineGraphRuntime *graph_runtime,
                           PipelineControlRuntime *control_runtime, const PlotLoopOptions &options) {
    const auto backend = options.backend == PlotBackend::Auto ? PlotBackend::OpenGL3 : options.backend;
    if (backend == PlotBackend::Vulkan) {
        throw std::runtime_error("the Vulkan plot backend is not built in Phase 22");
    }

    std::size_t initial_session_count = 0;
    {
        const auto lock = std::scoped_lock(runtime.mutex());
        initial_session_count = runtime.trace_view()->trace_snapshots().size();
    }

    log::LogRecord start_record{
        .level = log::LogLevel::Info,
        .component = "plot",
        .message = "starting plot runtime",
        .fields =
            {
                {"backend", std::string(to_string(backend))},
                {"sessions", std::to_string(initial_session_count)},
            },
    };
    log::write(std::move(start_record));

    GlfwContext glfw_context;
    GlfwWindow window(options.width, options.height, options.window_title);
    ImGuiContextOwner imgui_context(window.window);

    std::array<char, 512> png_path{};
    const auto default_path = std::string("leakflow_traceplot.png");
    std::copy(default_path.begin(), default_path.end(), png_path.begin());

    bool screenshot_requested = false;
    std::string screenshot_status;

    while (glfwWindowShouldClose(window.window) == 0) {
        if (options.should_close && options.should_close()) {
            glfwSetWindowShouldClose(window.window, GLFW_TRUE);
            break;
        }
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (options.show_controls_window) {
            const auto toolbar_title = options.window_title + std::string(" Controls");
            ImGui::Begin(toolbar_title.c_str());
            ImGui::InputText("PNG path", png_path.data(), png_path.size());
            ImGui::SameLine();
            if (ImGui::Button("Save PNG")) {
                screenshot_requested = true;
            }
            if (!screenshot_status.empty()) {
                ImGui::TextUnformatted(screenshot_status.c_str());
            }
            // Player controls (Stopped/Running/Paused/Idle). The UI calls the request;
            // the worker drives the state machine. Buttons enable/disable by state.
            if (control_runtime != nullptr && control_runtime->session() != nullptr) {
                ImGui::Separator();
                const auto state = control_runtime->session()->state();
                const bool stopped = state == leakflow::PipelineSessionState::Stopped;
                const bool running = state == leakflow::PipelineSessionState::Running;
                const bool paused = state == leakflow::PipelineSessionState::Paused;
                const bool idle = state == leakflow::PipelineSessionState::Idle;
                const char *label = stopped ? "Stopped" : running ? "Running" : paused ? "Paused" : "Idle";

                const auto gated_button = [](const char *text, bool enabled) {
                    ImGui::BeginDisabled(!enabled);
                    const bool clicked = ImGui::Button(text);
                    ImGui::EndDisabled();
                    return clicked;
                };

                ImGui::TextUnformatted(label);
                ImGui::SameLine();
                if (gated_button("Start", stopped)) { // Idle must Stop first; Stop recycles to a fresh run
                    control_runtime->request_start();
                }
                ImGui::SameLine();
                if (gated_button("Stop", running || paused || idle)) {
                    control_runtime->request_stop();
                }
                ImGui::SameLine();
                if (gated_button("Pause", running)) {
                    control_runtime->request_pause();
                }
                ImGui::SameLine();
                if (gated_button("Resume", paused)) {
                    control_runtime->request_resume();
                }

                // Auto-apply vs manual is orthogonal to the state: in manual mode edits
                // stage until Apply, in any state (Stopped/Running/Paused/Idle).
                bool auto_recompute = control_runtime->auto_recompute();
                if (ImGui::Checkbox("Auto-apply edits", &auto_recompute)) {
                    control_runtime->set_auto_recompute(auto_recompute);
                }
                if (!auto_recompute) {
                    ImGui::SameLine();
                    if (ImGui::Button("Apply")) {
                        control_runtime->request_apply();
                    }
                }
            }
            ImGui::End();
        }
        if (graph_runtime != nullptr) {
            draw_pipeline_graph(*graph_runtime, control_runtime);
        } else if (control_runtime != nullptr) {
            draw_pipeline_controls(*control_runtime);
        }
        if (graph_runtime == nullptr || runtime.has_sessions()) {
            draw_plot_runtime(runtime, control_runtime);
        }

        ImGui::Render();

        int display_width = 0;
        int display_height = 0;
        glfwGetFramebufferSize(window.window, &display_width, &display_height);
        glViewport(0, 0, display_width, display_height);
        glClearColor(0.09F, 0.10F, 0.11F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (screenshot_requested) {
            try {
                save_current_framebuffer_png(png_path.data(), display_width, display_height);
                screenshot_status = std::string("saved ") + png_path.data();
                log::LogRecord record{
                    .level = log::LogLevel::Info,
                    .component = "plot",
                    .message = "saved plot PNG",
                    .fields =
                        {
                            {"path", png_path.data()},
                        },
                };
                log::write(std::move(record));
            } catch (const std::exception &error) {
                screenshot_status = std::string("save failed: ") + error.what();
                log::LogRecord record{
                    .level = log::LogLevel::Error,
                    .component = "plot",
                    .message = "failed to save plot PNG",
                    .fields =
                        {
                            {"error", error.what()},
                        },
                };
                log::write(std::move(record));
            }
            screenshot_requested = false;
        }

        glfwSwapBuffers(window.window);
    }

    log::LogRecord finish_record{
        .level = log::LogLevel::Info,
        .component = "plot",
        .message = "stopped plot runtime",
    };
    log::write(std::move(finish_record));
}

void run_until_closed(PlotRuntime &runtime, PipelineGraphRuntime &graph_runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, &graph_runtime, nullptr, options);
}

void run_until_closed(PlotRuntime &runtime, PipelineGraphRuntime &graph_runtime,
                      PipelineControlRuntime &control_runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, &graph_runtime, &control_runtime, options);
}

void run_until_closed(PlotRuntime &runtime, PipelineControlRuntime &control_runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, nullptr, &control_runtime, options);
}

void run_until_closed(PlotRuntime &runtime, const PlotLoopOptions &options) {
    run_until_closed_impl(runtime, nullptr, nullptr, options);
}

} // namespace leakflow::plot
