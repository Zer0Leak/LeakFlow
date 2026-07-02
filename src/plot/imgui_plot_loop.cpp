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
#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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


// A "Windows" focus list for the controls toolbar: it enumerates the visible
// top-level ImGui windows (plots, graph, controls) and focuses one on click -- a
// reliable alternative to Ctrl+Tab, which depends on OS keyboard focus and modifier
// delivery (flaky under some Wayland / focus-follows-mouse setups).
void draw_window_focus_list() {
    ImGuiContext &g = *ImGui::GetCurrentContext();
    std::vector<ImGuiWindow *> targets;
    for (ImGuiWindow *candidate : g.Windows) {
        if (candidate == nullptr || !candidate->WasActive || candidate->ParentWindow != nullptr) {
            continue;
        }
        if ((candidate->Flags & (ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_Popup)) !=
            0) {
            continue;
        }
        targets.push_back(candidate);
    }
    if (targets.empty()) {
        return;
    }

    ImGui::TextUnformatted("Focus window:");
    const auto right_edge = ImGui::GetWindowContentRegionMax().x;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        ImGuiWindow *target = targets[index];
        if (index != 0) {
            const auto button_width = ImGui::CalcTextSize(target->Name, nullptr, true).x +
                                      g.Style.FramePadding.x * 2.0F + g.Style.ItemSpacing.x;
            if (ImGui::GetCursorPosX() + button_width < right_edge) {
                ImGui::SameLine();
            }
        }
        if (ImGui::SmallButton(target->Name)) {
            ImGui::SetWindowFocus(target->Name);
            if (target->Collapsed) {
                ImGui::SetWindowCollapsed(target->Name, false, ImGuiCond_Always);
            }
        }
    }
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
            draw_window_focus_list();
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
