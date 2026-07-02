#include "leakflow/core/pipeline_session.hpp"

#include "leakflow/log/logger.hpp"

#include <chrono>
#include <exception>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace leakflow {
namespace {

PipelineEndpointSnapshot endpoint_for(const std::shared_ptr<Element> &element, const std::string &fallback_name)
{
    PipelineEndpointSnapshot endpoint;
    if (element) {
        endpoint.element_type = element->element_type();
        endpoint.element_name = element->name();
        endpoint.element_klass = element->element_kclass();
    } else {
        endpoint.element_name = fallback_name;
    }
    return endpoint;
}

PipelineEventKind kind_for_status(PipelineCommandStatus status)
{
    switch (status) {
    case PipelineCommandStatus::Accepted:
        return PipelineEventKind::CommandAccepted;
    case PipelineCommandStatus::Rejected:
        return PipelineEventKind::CommandRejected;
    case PipelineCommandStatus::Applied:
    case PipelineCommandStatus::Failed:
        return PipelineEventKind::CommandApplied;
    }
    return PipelineEventKind::CommandApplied;
}

bool is_dataflow_effect(PropertyEffectKind kind)
{
    return kind == PropertyEffectKind::MetadataOutput || kind == PropertyEffectKind::PayloadOutput
        || kind == PropertyEffectKind::CapsOutput || kind == PropertyEffectKind::Lifecycle;
}

} // namespace

PipelineSession::PipelineSession(Pipeline pipeline)
    : pipeline_(std::move(pipeline))
{
}

Pipeline &PipelineSession::pipeline()
{
    return pipeline_;
}

const Pipeline &PipelineSession::pipeline() const
{
    return pipeline_;
}

void PipelineSession::set_observer(std::shared_ptr<PipelineObserver> observer)
{
    observer_ = observer;
    pipeline_.set_observer(std::move(observer));
}

std::shared_ptr<PipelineObserver> PipelineSession::observer() const
{
    return observer_;
}

void PipelineSession::set_telemetry_enabled(bool enabled)
{
    telemetry_enabled_ = enabled;
    pipeline_.set_runtime_telemetry_enabled(enabled);
}

bool PipelineSession::telemetry_enabled() const
{
    return telemetry_enabled_;
}

void PipelineSession::set_caching_enabled(bool enabled)
{
    pipeline_.set_caching_enabled(enabled);
}

bool PipelineSession::caching_enabled() const
{
    return pipeline_.caching_enabled();
}

void PipelineSession::set_stop_token(std::stop_token token)
{
    stop_token_ = token;
    pipeline_.set_stop_token(std::move(token));
}

void PipelineSession::set_safe_point_mutex(std::mutex *mutex)
{
    safe_point_mutex_ = mutex;
}

void PipelineSession::request_pause()
{
    paused_.store(true);
    auto expected = PipelineSessionState::Running;
    state_.compare_exchange_strong(expected, PipelineSessionState::Paused);
}

void PipelineSession::request_resume()
{
    auto expected = PipelineSessionState::Paused;
    state_.compare_exchange_strong(expected, PipelineSessionState::Running);
    {
        std::lock_guard<std::mutex> lock(pause_mutex_);
        paused_.store(false);
    }
    pause_cv_.notify_all();
}

bool PipelineSession::is_paused() const
{
    return paused_.load();
}

void PipelineSession::wait_while_paused(std::stop_token stop)
{
    if (!paused_.load()) {
        return;
    }
    std::unique_lock<std::mutex> lock(pause_mutex_);
    // Wakes on resume (predicate) or teardown (stop token); never blocks once
    // stop is requested, so a paused pipeline still tears down promptly.
    pause_cv_.wait(lock, std::move(stop), [this]() { return !paused_.load(); });
}

void PipelineSession::wait_paused_tick(std::stop_token stop)
{
    std::unique_lock<std::mutex> lock(pause_mutex_);
    pause_cv_.wait_for(lock, std::move(stop), std::chrono::milliseconds(50),
                       [this]() { return !paused_.load(); });
}

PipelineSessionState PipelineSession::state() const
{
    return state_.load();
}

void PipelineSession::set_state(PipelineSessionState state)
{
    state_.store(state);
}

std::uint64_t PipelineSession::generation() const
{
    return generation_;
}

void PipelineSession::start()
{
    if (state_.load() != PipelineSessionState::Stopped) {
        return;
    }
    pipeline_.start_all();
    state_ = PipelineSessionState::Running;
}

std::optional<Buffer> PipelineSession::run_sweep()
{
    if (state_.load() == PipelineSessionState::Stopped) {
        start();
    }
    auto output = pipeline_.run_sweep();
    state_ = PipelineSessionState::Running;
    return output;
}

std::optional<Buffer> PipelineSession::run_once()
{
    // Live + Queue: run threaded segments (step 4b). run_threaded manages its own
    // start_all/stop_all and asks boundary elements to own the handoff runtime,
    // blocking until every segment reaches end-of-stream or a cooperative stop is
    // requested. The headless front door has no mid-run control plane, so running to
    // completion is the whole job.
    if (pipeline_.should_run_threaded()) {
        state_ = PipelineSessionState::Running;
        forward_only_apply_ = true;
        std::optional<Buffer> output;
        try {
            // Each segment thread applies its own pending commands at a between-buffer
            // safe point, so live edits forward-apply mid-stream (S11.5).
            output = pipeline_.run_threaded(
                stop_token_, [this](const std::vector<std::shared_ptr<Element>> &elements, std::stop_token stop) {
                    // Apply pending edits now; while paused, keep applying edits made
                    // during the pause so they are SET on the element immediately
                    // (Auto mode) -- the stream stays frozen, the visible result
                    // updates on Resume. (Manual mode stages edits, so nothing here
                    // applies until Apply flushes them: batch-on-resume.)
                    apply_commands_for(elements);
                    while (is_paused() && !stop.stop_requested()) {
                        wait_paused_tick(stop);
                        apply_commands_for(elements);
                    }
                },
                safe_point_mutex_, telemetry_enabled_);
        } catch (...) {
            forward_only_apply_ = false;
            state_ = PipelineSessionState::Stopped;
            throw;
        }
        forward_only_apply_ = false;
        state_ = PipelineSessionState::Stopped;
        return output;
    }

    start();
    std::optional<Buffer> output;
    if (pipeline_.has_live_source()) {
        // Live: pump until every live source reaches end-of-stream, or a
        // cooperative stop is requested (Ctrl+C / window-close) (S11.8).
        while (!pipeline_.all_live_sources_at_eos() && !pipeline_.stop_requested()) {
            output = pipeline_.run_sweep();
        }
    } else {
        output = pipeline_.run_sweep();
    }
    stop();
    return output;
}

void PipelineSession::stop()
{
    if (state_.load() != PipelineSessionState::Stopped) {
        pipeline_.stop_all();
    }
    state_ = PipelineSessionState::Stopped;
}

void PipelineSession::restart()
{
    pipeline_.stop_all();
    pipeline_.start_all();
    ++generation_;
    (void)pipeline_.run_sweep();
    state_ = PipelineSessionState::Running;
}

std::optional<Buffer> PipelineSession::rerun_from_sources()
{
    if (state_.load() == PipelineSessionState::Stopped) {
        start();
    }
    ++generation_;
    auto output = pipeline_.run_sweep();
    state_ = PipelineSessionState::Running;
    return output;
}

namespace {

void merge_command(std::vector<SetPropertyCommand> &queue, SetPropertyCommand command)
{
    for (auto &existing : queue) {
        if (existing.element_name == command.element_name && existing.property_name == command.property_name) {
            existing.value = std::move(command.value); // last-wins per (element, property)
            return;
        }
    }
    queue.push_back(std::move(command));
}

} // namespace

void PipelineSession::submit(SetPropertyCommand command)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    // Manual mode stages the edit; otherwise it queues live for immediate apply.
    merge_command(manual_apply_ ? staged_ : pending_, std::move(command));
}

void PipelineSession::set_manual_apply(bool on)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    manual_apply_ = on;
}

void PipelineSession::apply_staged()
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    for (auto &command : staged_) {
        merge_command(pending_, std::move(command));
    }
    staged_.clear();
}

std::size_t PipelineSession::drain_commands()
{
    std::vector<SetPropertyCommand> batch;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        batch.swap(pending_);
    }

    for (const auto &command : batch) {
        apply_command(command);
    }
    return batch.size();
}

std::size_t PipelineSession::pending_command_count() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return pending_.size();
}

std::shared_ptr<Element>
PipelineSession::first_non_replayable_in_replay_set(const std::shared_ptr<Element> &element) const
{
    for (const auto &member : pipeline_.replay_set(element)) {
        if (!member->can_replay()) {
            return member;
        }
    }
    return nullptr;
}

void PipelineSession::emit_command(PipelineCommandStatus status, const std::shared_ptr<Element> &element,
                                   const SetPropertyCommand &command, const PropertyEffect &effect,
                                   std::string previous_value, std::string detail)
{
    PipelineCommandObservation observation{
        .status = status,
        .element = endpoint_for(element, command.element_name),
        .property_name = command.property_name,
        .value_type = property_value_type_name(command.value),
        .previous_value = std::move(previous_value),
        .new_value = property_value_to_string(command.value),
        .effect = effect,
        .generation = generation_,
        .detail = std::move(detail),
    };
    emit(PipelineEvent{
        .kind = kind_for_status(status),
        .command = std::move(observation),
    });
}

void PipelineSession::emit(PipelineEvent event)
{
    if (!observer_) {
        return;
    }
    event.sequence = next_event_sequence_++;
    try {
        observer_->observe(event);
    } catch (...) {
    }
}

void PipelineSession::apply_commands_for(const std::vector<std::shared_ptr<Element>> &elements)
{
    if (elements.empty()) {
        return;
    }
    std::set<std::string> names;
    for (const auto &element : elements) {
        names.insert(element->name());
    }

    // Atomically remove the commands for this segment's elements from the shared
    // queue (last-wins per element/property is preserved by submit()).
    std::vector<SetPropertyCommand> mine;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::vector<SetPropertyCommand> rest;
        rest.reserve(pending_.size());
        for (auto &command : pending_) {
            if (names.contains(command.element_name)) {
                mine.push_back(std::move(command));
            } else {
                rest.push_back(std::move(command));
            }
        }
        pending_.swap(rest);
    }

    if (mine.empty()) {
        return;
    }

    // Serialize against a concurrent UI reader (the graph) only when there is
    // actually something to apply, so the common no-command safe point is lock-free.
    std::unique_lock<std::mutex> ui_lock;
    if (safe_point_mutex_ != nullptr) {
        ui_lock = std::unique_lock<std::mutex>(*safe_point_mutex_);
    }
    for (const auto &command : mine) {
        apply_command(command);
    }
}

void PipelineSession::apply_command(const SetPropertyCommand &command)
{
    // Serialize session-state mutations (generation_, event sequence, observer)
    // against other segment threads applying commands concurrently. Per-element
    // property writes are already single-thread (the element's owning segment).
    std::lock_guard<std::mutex> control_lock(control_mutex_);

    const PropertyEffect no_effect{};
    auto element = pipeline_.find_element(command.element_name);
    if (!element) {
        emit_command(PipelineCommandStatus::Rejected, nullptr, command, no_effect, std::string(), "unknown element");
        return;
    }
    if (!element->has_property(command.property_name)) {
        emit_command(PipelineCommandStatus::Rejected, element, command, no_effect, std::string(), "unknown property");
        return;
    }

    const PropertySpec *property_spec = nullptr;
    for (const auto &spec : element->property_specs()) {
        if (spec.name == command.property_name) {
            property_spec = &spec;
            break;
        }
    }
    if (property_spec == nullptr) {
        emit_command(PipelineCommandStatus::Rejected, element, command, no_effect, std::string(), "unknown property");
        return;
    }
    const auto effect = property_spec->effect;

    const PropertyValue previous_property = element->property(command.property_name);
    std::string previous_value = property_value_to_string(previous_property);

    try {
        element->validate_property_change(command.property_name, command.value);
    } catch (const std::exception &error) {
        emit_command(PipelineCommandStatus::Rejected, element, command, effect, std::move(previous_value),
                     error.what());
        return;
    }

    const auto current_state = state_.load();
    const bool live_driven = pipeline_.is_live_driven(element);
    const bool restart_scoped = effect.kind == PropertyEffectKind::Lifecycle
        || effect.scope == PropertyInvalidationScope::FullPipeline;
    const bool data_or_display_effect =
        effect.kind == PropertyEffectKind::SinkDisplay || is_dataflow_effect(effect.kind);
    const bool idle = current_state == PipelineSessionState::Idle;
    const bool would_reprocess_existing_data =
        current_state != PipelineSessionState::Stopped
        && !forward_only_apply_
        && !restart_scoped
        && data_or_display_effect
        && (!live_driven || idle);
    if (would_reprocess_existing_data) {
        if (const auto blocker = first_non_replayable_in_replay_set(element)) {
            auto detail = std::string("property change rejected: it would reprocess existing buffers through "
                                      "non-replayable element '");
            detail += blocker->name();
            detail += "'; stop the pipeline or switch that element to a replayable mode";
            emit_command(PipelineCommandStatus::Rejected, element, command, effect, std::move(previous_value),
                         std::move(detail));
            return;
        }
    }

    try {
        element->set_property(command.property_name, command.value);
    } catch (const std::exception &error) {
        emit_command(PipelineCommandStatus::Rejected, element, command, effect, std::move(previous_value),
                     error.what());
        return;
    }

    // D4: caps-output changes are validated against downstream links before they
    // are allowed to stand. Revert and reject if any link would break.
    if (effect.kind == PropertyEffectKind::CapsOutput) {
        if (auto link_error = pipeline_.link_caps_error()) {
            element->set_property(command.property_name, previous_property);
            emit_command(PipelineCommandStatus::Rejected, element, command, effect, std::move(previous_value),
                         std::move(*link_error));
            return;
        }
    }

    emit_command(PipelineCommandStatus::Accepted, element, command, effect, previous_value, std::string());

    // Property-change observation for UI parity with the previous control path.
    emit(PipelineEvent{
        .kind = PipelineEventKind::PropertyChanged,
        .property_change =
            PipelinePropertyChangeObservation{
                .element = endpoint_for(element, command.element_name),
                .property_name = command.property_name,
                .value_type = property_value_type_name(command.value),
                .previous_value = previous_value,
                .new_value = property_value_to_string(command.value),
                .effect = effect,
            },
    });

    if (effect.kind == PropertyEffectKind::UiControl) {
        // Non-dataflow control/presentation: the element already applied the change
        // to its own UI/display state in property_changed (above, via set_property).
        // Nothing to rerun, no cache, no new generation; valid in any player state.
        emit_command(PipelineCommandStatus::Applied, element, command, effect, std::move(previous_value),
                     "no rerun");
        return;
    }

    if (effect.kind == PropertyEffectKind::SinkDisplay) {
        // Refresh this element's sink/display state by reprocessing it (and any
        // downstream) from cached inputs. No new configuration epoch: a display
        // change is not a new data generation.
        try {
            // forward_only_apply_: a threaded run streams continuously, so never
            // rerun -- the next buffer reprocesses with the new value (S11.5).
            if (current_state != PipelineSessionState::Stopped && !forward_only_apply_) {
                if (pipeline_.caching_enabled()) {
                    (void)pipeline_.rerun_from(element);
                } else {
                    (void)pipeline_.run_sweep();
                }
            }
        } catch (const std::exception &error) {
            emit_command(PipelineCommandStatus::Failed, element, command, effect, std::move(previous_value),
                         error.what());
            return;
        }
        emit_command(PipelineCommandStatus::Applied, element, command, effect, std::move(previous_value),
                     "sink-display");
        return;
    }

    // Dataflow change: a new configuration generation begins.
    ++generation_;

    // Liveness-aware property change (S11.5): on a LIVE-driven element the change
    // applies FORWARD -- the next pumped buffer naturally uses the new config -- so
    // we do NOT re-emit from cache. On a one-run-driven element we re-emit from
    // cache (the historical behavior), because no future buffer is coming.
    const bool force_restart = restart_scoped;

    {
        auto record = element->make_log_record(log::LogLevel::Info, "session", "applying property change");
        record.fields.emplace("property", command.property_name);
        record.fields.emplace("effect", std::string(property_effect_kind_name(effect.kind)));
        record.fields.emplace("generation", std::to_string(generation_));
        const bool forward_live =
            live_driven && current_state != PipelineSessionState::Idle
            && current_state != PipelineSessionState::Stopped;
        record.fields.emplace("mode", forward_live ? "forward-live" : (force_restart ? "full-restart" : "downstream"));
        leakflow::log::write(std::move(record));
    }

    try {
        if (current_state == PipelineSessionState::Stopped) {
            // No live run to update; the next run_sweep will pick up the change.
        } else if ((live_driven && current_state != PipelineSessionState::Idle) || forward_only_apply_) {
            // Forward: the change applies to the next pumped buffer. No cache
            // re-emit, no stall (S11.5). In Idle, there is no next live buffer, so
            // live-driven edits replay cached data like one-run edits and must pass
            // the replay-safety check above. During a threaded run
            // (forward_only_apply_) every change is forward -- segments stream with
            // no rerun point.
        } else if (force_restart) {
            pipeline_.stop_all();
            pipeline_.start_all();
            (void)pipeline_.run_sweep();
        } else if (pipeline_.caching_enabled()) {
            (void)pipeline_.rerun_from(element);
        } else {
            (void)pipeline_.run_sweep();
        }
    } catch (const std::exception &error) {
        // D12: a failed rerun does not tear down the session.
        emit_command(PipelineCommandStatus::Failed, element, command, effect, std::move(previous_value),
                     error.what());
        return;
    }

    emit_command(PipelineCommandStatus::Applied, element, command, effect, std::move(previous_value), std::string());
}

} // namespace leakflow
