#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/pipeline.hpp"
#include "leakflow/core/pipeline_observer.hpp"
#include "leakflow/core/property.hpp"

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

enum class PipelineSessionState {
    Stopped,
    Started,
    Running,
    Paused, // reserved for StreamingDrive; not entered in Phase 25
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

    [[nodiscard]] PipelineSessionState state() const;
    // Control-plane configuration-generation counter (Phase 27, renamed from
    // epoch). Per-buffer generation now lives in the vector clock; this is
    // session bookkeeping surfaced in command observations.
    [[nodiscard]] std::uint64_t generation() const;

    // Lifecycle / session controls (worker thread).
    void start();                                  // Stopped -> Started
    [[nodiscard]] std::optional<Buffer> run_sweep(); // Started/Running -> Running, full sweep
    [[nodiscard]] std::optional<Buffer> run_once();  // headless: start -> sweep -> stop
    void stop();                                   // graceful stop -> Stopped
    void restart();                                // stop -> start -> sweep (stop-start cycle), new generation
    [[nodiscard]] std::optional<Buffer> rerun_from_sources(); // full sweep, elements stay alive, new generation

    // Control plane.
    void submit(SetPropertyCommand command); // thread-safe; last-wins per (element, property)
    std::size_t drain_commands();            // worker thread; applies queued commands at a safe point
    [[nodiscard]] std::size_t pending_command_count() const;

private:
    void apply_command(const SetPropertyCommand &command);
    [[nodiscard]] bool replay_set_is_replayable(const std::shared_ptr<Element> &element) const;
    void emit_command(PipelineCommandStatus status, const std::shared_ptr<Element> &element,
                      const SetPropertyCommand &command, const PropertyEffect &effect, std::string previous_value,
                      std::string detail);
    void emit(PipelineEvent event);

    Pipeline pipeline_;
    std::shared_ptr<PipelineObserver> observer_;
    PipelineSessionState state_ = PipelineSessionState::Stopped;
    std::uint64_t generation_ = 1;
    std::uint64_t next_event_sequence_ = 1;

    mutable std::mutex queue_mutex_;
    std::vector<SetPropertyCommand> pending_;
    std::stop_token stop_token_;
};

} // namespace leakflow
