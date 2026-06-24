#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_observer.hpp"
#include "leakflow/core/property.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

// Control/session layer (Phase 25). Owns a Pipeline and is the
// application's single observe-and-control handle. Authoritative design:
// docs/design/pipeline_controller.md.
//
// Threading contract: every method except submit() is intended to be called from
// the single worker/owner thread (or single-threaded in headless use). submit()
// is thread-safe so a UI thread can enqueue SetProperty commands while the worker
// drains and applies them at a safe point via drain_commands().

// Streaming lifecycle (GStreamer-inspired). Stopped = torn down (no cache);
// Running = producing; Paused = frozen but resumable (a live source still has more);
// Idle = run finished and held (single run done, or the live source consumed all) --
// not resumable, but the cache is kept so edits can recompute (offline) or stage
// (live). Paused vs Idle differ only by whether there is more to stream.
enum class PipelineSessionState {
    Stopped,
    Running,
    Paused,
    Idle,
};

struct SetPropertyCommand {
    std::string element_name;
    std::string property_name;
    PropertyValue value;
};

class PipelineSession {
public:
    explicit PipelineSession(Pipeline pipeline);

    [[nodiscard]] Pipeline &pipeline();
    [[nodiscard]] const Pipeline &pipeline() const;

    void set_observer(std::shared_ptr<PipelineObserver> observer);
    [[nodiscard]] std::shared_ptr<PipelineObserver> observer() const;

    void set_caching_enabled(bool enabled);
    [[nodiscard]] bool caching_enabled() const;

    // Cooperative stop (live phase, S11.8). Forwarded to the pipeline so a live
    // pump (run_once / graph worker) exits promptly and blocking element waits are
    // interruptible on Ctrl+C / window-close.
    void set_stop_token(std::stop_token token);

    // Optional mutex held while applying commands at a threaded-run safe point
    // (S11.5). A UI that reads live element state concurrently (the graph's control
    // windows) passes the same mutex so its reads serialize against these writes.
    // Headless (no concurrent reader) leaves it null. Not owned.
    void set_safe_point_mutex(std::mutex *mutex);

    // Pause / resume a running threaded pipeline (Stopped/Running/Paused/Idle model).
    // request_pause parks every segment at its next between-buffer safe point (the
    // source stops producing, so the whole pipeline freezes); request_resume wakes
    // them. Thread-safe; called from the UI thread while segments run. The stop
    // token unblocks a parked segment on teardown.
    void request_pause();
    void request_resume();
    [[nodiscard]] bool is_paused() const;
    void wait_while_paused(std::stop_token stop);

    [[nodiscard]] PipelineSessionState state() const;
    // Worker-facing: drive the streaming lifecycle (Running on a run start, Idle when
    // a run finishes held, Stopped on teardown). Thread-safe (atomic).
    void set_state(PipelineSessionState state);
    // Control-plane configuration-generation counter (Phase 27, renamed from
    // epoch). Per-buffer generation now lives in the vector clock; this is
    // session bookkeeping surfaced in command observations.
    [[nodiscard]] std::uint64_t generation() const;

    // Lifecycle / session controls (worker thread).
    void start();                                  // Stopped -> Running (start_all)
    [[nodiscard]] std::optional<Buffer> run_sweep(); // -> Running, full sweep
    [[nodiscard]] std::optional<Buffer> run_once();  // headless: start -> sweep -> stop
    void stop();                                   // graceful stop -> Stopped
    void restart();                                // stop -> start -> sweep (stop-start cycle), new generation
    [[nodiscard]] std::optional<Buffer> rerun_from_sources(); // full sweep, elements stay alive, new generation

    // Control plane.
    void submit(SetPropertyCommand command); // thread-safe; last-wins per (element, property)
    std::size_t drain_commands();            // worker thread; applies queued commands at a safe point
    [[nodiscard]] std::size_t pending_command_count() const;

    // Manual-apply mode (orthogonal to the streaming state). When on, submit() stages
    // edits instead of queuing them live; apply_staged() flushes the staged edits into
    // the live queue (so the worker/segments then apply them as usual). This lets the
    // UI batch several edits and apply them together, in ANY state. Default off
    // (every edit is queued immediately).
    void set_manual_apply(bool on);
    void apply_staged();

private:
    // Safe-point callback for the threaded runner (S11.5): apply pending commands
    // targeting `elements` on the calling segment thread, forward-only (no rerun).
    void apply_commands_for(const std::vector<std::shared_ptr<Element>> &elements);
    void apply_command(const SetPropertyCommand &command);
    [[nodiscard]] bool replay_set_is_replayable(const std::shared_ptr<Element> &element) const;
    void emit_command(PipelineCommandStatus status, const std::shared_ptr<Element> &element,
                      const SetPropertyCommand &command, const PropertyEffect &effect, std::string previous_value,
                      std::string detail);
    void emit(PipelineEvent event);

    Pipeline pipeline_;
    std::shared_ptr<PipelineObserver> observer_;
    std::atomic<PipelineSessionState> state_ = PipelineSessionState::Stopped;
    std::uint64_t generation_ = 1;
    std::uint64_t next_event_sequence_ = 1;

    mutable std::mutex queue_mutex_;
    std::vector<SetPropertyCommand> pending_;
    // Manual-apply staging buffer; edits wait here until apply_staged() flushes them
    // into pending_. Guarded by queue_mutex_ alongside pending_.
    std::vector<SetPropertyCommand> staged_;
    bool manual_apply_ = false;
    std::stop_token stop_token_;

    // Serializes apply_command across segment threads during a threaded run (it
    // touches generation_ / event sequence). Property writes themselves are
    // single-thread-per-element (the owning segment), so no per-element lock.
    std::mutex control_mutex_;
    // While a threaded run is active, every command applies forward-only (no rerun):
    // segments stream continuously, so the change lands on the next buffer (S11.5).
    bool forward_only_apply_ = false;
    // Serializes safe-point command application against a concurrent UI reader (the
    // graph). Null when there is no such reader (headless). Not owned.
    std::mutex *safe_point_mutex_ = nullptr;

    // Pause primitive: segment threads park in wait_while_paused() while paused_.
    std::atomic<bool> paused_ = false;
    std::mutex pause_mutex_;
    std::condition_variable_any pause_cv_;
};

} // namespace leakflow
