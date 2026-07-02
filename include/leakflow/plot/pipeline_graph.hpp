#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/pipeline_observer.hpp"
#include "leakflow/core/property.hpp"
#include "leakflow/plot/plot_runtime.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

class Element;
class Pipeline;
class PipelineSession;

} // namespace leakflow

namespace leakflow::plot {

class PipelineGraphRuntime final : public PipelineObserver {
public:
    void observe(const PipelineEvent &event) override;
    void drain_events();
    void clear();

    [[nodiscard]] bool has_topology() const;
    [[nodiscard]] bool running() const;
    [[nodiscard]] bool stopped() const;
    [[nodiscard]] const PipelineTopologySnapshot &topology() const;
    [[nodiscard]] const std::map<std::string, PipelineBufferObservation> &latest_buffers() const;
    [[nodiscard]] std::uint64_t observed_count(std::string_view link_id) const;
    [[nodiscard]] const std::optional<std::string> &last_error() const;
    [[nodiscard]] const std::vector<PipelineEvent> &recent_events() const;

    // Vector-clock provenance (Phase 27): the running component-wise max over all
    // observed buffers, i.e. the current production count per slot index.
    [[nodiscard]] const std::vector<std::uint32_t> &max_provenance() const;

    // Pinned info panels (UI state): clicking a node/link pins a collapsible,
    // mouse-interactable window so the user can read/collapse it without the
    // hover tooltip vanishing.
    void toggle_pinned_element(std::string_view name);
    [[nodiscard]] bool is_element_pinned(std::string_view name) const;
    [[nodiscard]] const std::set<std::string> &pinned_elements() const;
    void unpin_element(std::string_view name);
    void toggle_pinned_link(std::string_view link_id);
    [[nodiscard]] bool is_link_pinned(std::string_view link_id) const;
    [[nodiscard]] const std::set<std::string> &pinned_links() const;
    void unpin_link(std::string_view link_id);

    // Shared collapse state for info-panel sections, keyed by section label, so
    // the interactive pinned panel and the non-interactive hover tooltip agree on
    // which sections are open. The tooltip mirrors what the panel chose.
    [[nodiscard]] bool section_open(std::string_view label, bool default_open) const;
    void set_section_open(std::string_view label, bool open);

private:
    void apply_event(const PipelineEvent &event);

    mutable std::mutex pending_mutex_;
    std::vector<PipelineEvent> pending_events_;

    PipelineTopologySnapshot topology_;
    bool has_topology_ = false;
    bool running_ = false;
    bool stopped_ = false;
    std::map<std::string, PipelineBufferObservation> latest_buffers_;
    std::map<std::string, std::uint64_t> observed_counts_;
    std::map<std::string, std::uint32_t> link_generations_;
    std::vector<std::uint32_t> max_provenance_;
    std::set<std::string> pinned_elements_;
    std::set<std::string> pinned_links_;
    std::map<std::string, bool> section_open_;
    std::optional<std::string> last_error_;
    std::vector<PipelineEvent> recent_events_;
};

struct PipelineControlChange {
    std::string element_name;
    std::string property_name;
    std::string previous_value;
    std::string new_value;
    PropertyEffect effect;
};

class PipelineControlRuntime final {
public:
    struct TextEditState {
        std::string text;
        std::string observed_value;
        bool dirty = false;
        bool initialized = false;
    };

    // Persistent edit buffer for the color picker so its popup is not clobbered
    // by the async (command-queue) value while the user is choosing a color.
    struct ColorEditState {
        std::array<float, 4> rgba{};
        bool dirty = false;
    };

    void bind(Pipeline &pipeline);
    void bind(std::shared_ptr<Element> element);
    void clear();

    [[nodiscard]] bool has_element(std::string_view element_name) const;
    [[nodiscard]] std::vector<std::string> element_names() const;
    [[nodiscard]] std::shared_ptr<Element> element(std::string_view element_name) const;

    void open(std::string_view element_name);
    void close(std::string_view element_name);
    [[nodiscard]] bool is_open(std::string_view element_name) const;

    void set_edits_enabled(bool enabled);
    [[nodiscard]] bool edits_enabled() const;

    void set_observer(std::shared_ptr<PipelineObserver> observer);
    [[nodiscard]] std::shared_ptr<PipelineObserver> observer() const;

    // Session client wiring (Phase 25). When a session is bound, set_property
    // submits a command to the session instead of mutating the live element, and
    // the control windows take element_mutex while reading live element state so
    // they never race the worker thread applying commands.
    void bind_session(PipelineSession *session);
    [[nodiscard]] PipelineSession *session() const;
    void set_element_mutex(std::mutex *element_mutex);
    [[nodiscard]] std::mutex *element_mutex() const;

    // Player controls (Stopped/Running/Paused/Idle). The UI thread calls these;
    // Start/Apply set a flag the worker polls, Stop invokes the worker-registered
    // run-stopper immediately (to interrupt a blocking run), and Pause/Resume forward
    // straight to the session's pause primitive.
    void request_start();
    void request_stop();
    void request_pause();
    void request_resume();
    void request_apply(); // flush staged edits (manual mode); orthogonal to state
    [[nodiscard]] bool take_start_request();
    [[nodiscard]] bool take_user_stopped();
    void set_run_stopper(std::function<void()> stopper); // worker registers how to stop the current run
    void set_auto_recompute(bool on);
    [[nodiscard]] bool auto_recompute() const;

    [[nodiscard]] bool set_property(std::string_view element_name, std::string_view property_name,
                                    PropertyValue value);
    [[nodiscard]] std::vector<PipelineControlChange> take_changes();
    [[nodiscard]] const std::optional<std::string> &last_error() const;
    [[nodiscard]] std::vector<std::string> open_element_names() const;

    [[nodiscard]] TextEditState &text_edit_state(std::string_view key);
    [[nodiscard]] ColorEditState &color_edit_state(std::string_view key);
    void set_last_error(std::string error);

private:
    friend void draw_pipeline_controls(PipelineControlRuntime &runtime);
    friend void draw_pipeline_graph(PipelineGraphRuntime &runtime, PipelineControlRuntime *control_runtime);

    std::map<std::string, std::weak_ptr<Element>> elements_;
    std::set<std::string> open_elements_;
    std::map<std::string, TextEditState> text_edit_states_;
    std::map<std::string, ColorEditState> color_edit_states_;
    std::vector<PipelineControlChange> changes_;
    std::optional<std::string> last_error_;
    std::shared_ptr<PipelineObserver> observer_;
    std::uint64_t next_event_sequence_ = 1;
    bool edits_enabled_ = true;
    PipelineSession *session_ = nullptr;
    std::mutex *element_mutex_ = nullptr;
    std::atomic<bool> start_requested_ = false;
    std::atomic<bool> user_stopped_ = false;
    std::atomic<bool> auto_recompute_ = true; // default: apply edits immediately
    std::mutex run_stopper_mutex_;
    std::function<void()> run_stopper_;
};

void draw_pipeline_graph(PipelineGraphRuntime &runtime);
void draw_pipeline_graph(PipelineGraphRuntime &runtime, PipelineControlRuntime *control_runtime);
void draw_pipeline_controls(PipelineControlRuntime &runtime);

[[nodiscard]] std::optional<Buffer> run_pipeline_graph_until_closed(
    Pipeline &pipeline,
    PlotRuntime &plot_runtime,
    const PlotLoopOptions &options = {});
[[nodiscard]] std::optional<Buffer> run_pipeline_graph_until_closed(
    Pipeline &pipeline,
    PlotRuntime &plot_runtime,
    PipelineControlRuntime &control_runtime,
    const PlotLoopOptions &options = {});

// Session-driven graph runner (Phase 25). Drives a PipelineSession with a
// persistent worker loop: start -> initial sweep -> drain queued SetProperty
// commands and session-control actions at safe points -> graceful stop on close.
// Control edits are submitted to the session; live element reads in the control
// windows are serialized against the worker via an internal mutex.
[[nodiscard]] std::optional<Buffer> run_pipeline_graph_until_closed(
    PipelineSession &session,
    PlotRuntime &plot_runtime,
    const PlotLoopOptions &options = {});
[[nodiscard]] std::optional<Buffer> run_pipeline_graph_until_closed(
    PipelineSession &session,
    PlotRuntime &plot_runtime,
    PipelineControlRuntime &control_runtime,
    const PlotLoopOptions &options = {});

} // namespace leakflow::plot
