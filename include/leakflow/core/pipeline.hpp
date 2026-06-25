#pragma once

#include "leakflow/core/element.hpp"
#include "leakflow/core/pipeline_observer.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

class BufferQueue;
struct PipelineSegment;

struct PadLink {
    std::shared_ptr<Element> source_element;
    std::string source_pad_name;
    std::shared_ptr<Element> sink_element;
    std::string sink_pad_name;
};

struct OutputMetadataAnnotation {
    std::shared_ptr<Element> source_element;
    MetadataPadTarget target;
    std::map<std::string, std::string> metadata;
};

class Pipeline {
public:
    std::shared_ptr<Element> add(std::shared_ptr<Element> element);
    void link(std::shared_ptr<Element> source_element, std::string source_pad_name,
              std::shared_ptr<Element> sink_element, std::string sink_pad_name);
    void add_output_metadata_annotation(std::shared_ptr<Element> source_element, std::string source_pad_name,
                                        std::map<std::string, std::string> metadata);
    void add_output_metadata_annotation(std::shared_ptr<Element> source_element,
                                        std::map<std::string, std::string> metadata);
    void add_output_metadata_annotation_for_pad_template(std::shared_ptr<Element> source_element,
                                                         std::string source_pad_template,
                                                         std::map<std::string, std::string> metadata);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] const std::vector<std::shared_ptr<Element>> &elements() const;
    [[nodiscard]] std::shared_ptr<Element> find_element(std::string_view name) const;
    [[nodiscard]] std::shared_ptr<Element> element(std::string_view name) const;
    [[nodiscard]] std::vector<std::shared_ptr<Element>> elements_by_type(std::string_view type_name) const;
    [[nodiscard]] const std::vector<PadLink> &links() const;
    [[nodiscard]] Caps source_caps(const Element &source_element, std::string_view source_pad_name) const;
    [[nodiscard]] PipelineTopologySnapshot topology_snapshot() const;

    // Revalidate caps compatibility for every link (Phase 25, D4). Returns a
    // human-readable error for the first incompatible link, or nullopt if all
    // links are still compatible. Used to reject caps-output property changes
    // transactionally.
    [[nodiscard]] std::optional<std::string> link_caps_error() const;

    void set_observer(std::shared_ptr<PipelineObserver> observer);
    [[nodiscard]] std::shared_ptr<PipelineObserver> observer() const;

    // Caching of routed buffers for partial rerun (Phase 25). Default on. When
    // off, rerun_from cannot be used and dataflow changes must full re-sweep.
    void set_caching_enabled(bool enabled);
    [[nodiscard]] bool caching_enabled() const;

    // Execution primitives (Phase 25). run() is sugar over start_all/run_sweep/
    // stop_all. The session drives the finer-grained lifecycle directly. Buffer
    // generation is carried by the vector clock (Phase 27), not an epoch arg.
    void start_all();
    [[nodiscard]] std::optional<Buffer> run_sweep();
    [[nodiscard]] std::optional<Buffer> rerun_from(const std::shared_ptr<Element> &element);
    void stop_all();

    // Forward-reachable elements from `element` (inclusive), in pipeline order.
    [[nodiscard]] std::vector<std::shared_ptr<Element>> replay_set(const std::shared_ptr<Element> &element) const;

    // Liveness propagation (live phase, S11.5): an element is "live-driven" if it
    // is a live source or any element with a path to it (via links) is live.
    // Drives the session's forward-vs-cache property-change behavior.
    [[nodiscard]] bool is_live_driven(const std::shared_ptr<Element> &element) const;

    // Live-streaming detection (live phase): whether the graph has any live source,
    // and whether all live sources have reached end-of-stream. Used by run() and by
    // PipelineSession to drive the pump loop vs the offline one-shot.
    [[nodiscard]] bool has_live_source() const;
    [[nodiscard]] bool all_live_sources_at_eos() const;

    // True when the live runner should thread: a live source AND at least one Queue
    // (so the graph decomposes into >1 segment). Offline pipelines and Queue-free
    // live pipelines stay on the single-threaded sweep path. Drives run() and the
    // session's run_once() (live phase, step 4b).
    [[nodiscard]] bool should_run_threaded() const;

    // Cooperative stop (live phase, S11.8). Forwards the token to every element
    // (so blocking waits become interruptible) and arms the run()/pump loop to
    // exit promptly once a stop is requested. Default (unset) token never stops.
    void set_stop_token(std::stop_token token);
    [[nodiscard]] bool stop_requested() const;

    [[nodiscard]] std::optional<Buffer> run();

    // Invoked by each segment thread at a between-buffer safe point with that
    // segment's elements and the run's stop token. The callee parks while paused and
    // applies pending property changes to those elements ON the segment thread, so a
    // change forward-applies to the next buffer with no data race (the same thread
    // reads the property in process()). See the session's safe-point control plane
    // (S11.5) and the pause primitive.
    using SegmentSafePoint = std::function<void(const std::vector<std::shared_ptr<Element>> &, std::stop_token)>;

    // Threaded live runner (live phase, S10/S11.8). Decomposes the pipeline into
    // segments (cut at every Queue) and runs one std::jthread per segment, with a
    // BufferQueue per Queue as the thread-safe cross-segment handoff. A Queue-free
    // pipeline is a single segment and runs on one streaming thread (equivalent to
    // run()). Honors the cooperative stop token; on a segment failure, peers are
    // stopped so none hang on a queue. Elements stay synchronous and thread-unaware.
    // The optional safe_point is called between buffers per segment (S11.5).
    [[nodiscard]] std::optional<Buffer> run_threaded(std::stop_token stop, SegmentSafePoint safe_point = {});

private:
    [[nodiscard]] std::vector<std::shared_ptr<Element>> linked_execution_order() const;
    [[nodiscard]] std::vector<PadLink> incoming_links_for(const std::shared_ptr<Element> &element) const;
    [[nodiscard]] std::vector<PadLink> outgoing_links_for(const std::shared_ptr<Element> &element) const;
    [[nodiscard]] std::optional<Buffer> execute(const std::vector<std::shared_ptr<Element>> &order,
                                                bool seed_from_cache);
    void refresh_live_driven_flags();

    // Threaded-runner helpers (live phase). A segment thread runs its elements
    // synchronously; a source-driven segment (no input queues) pumps until its live
    // sources reach EOS, a consumer segment pulls from its input queues. execute_segment
    // is the per-sweep gather/stamp/route, seeded from queue pulls, pushing boundary
    // outputs to BufferQueues. `shared` guards the pipeline's cross-thread counters.
    using QueueRuntimes = std::map<const Element *, std::shared_ptr<BufferQueue>>;
    std::optional<Buffer> run_source_segment(const PipelineSegment &segment, const QueueRuntimes &queues,
                                             std::stop_token stop, std::mutex &shared,
                                             const SegmentSafePoint &safe_point);
    std::optional<Buffer> run_consumer_segment(const PipelineSegment &segment, const QueueRuntimes &queues,
                                               std::stop_token stop, std::mutex &shared,
                                               const SegmentSafePoint &safe_point);
    std::optional<Buffer> execute_segment(const std::vector<std::shared_ptr<Element>> &order,
                                          std::map<const Element *, std::map<std::string, Buffer>> live,
                                          const QueueRuntimes &queues, std::stop_token stop, std::mutex &shared);

    void stop_started(std::size_t started_count) noexcept;
    void emit(PipelineEvent event) noexcept;

    // Vector-clock provenance (Phase 27): per-element base slot allocated at
    // add(), and a per-element monotonic emit count (wraps max->1, 0 reserved).
    [[nodiscard]] std::uint32_t next_emit_count(const Element *element);

    std::vector<std::shared_ptr<Element>> elements_;
    std::map<std::string, std::shared_ptr<Element>> elements_by_name_;
    std::vector<PadLink> links_;
    std::vector<OutputMetadataAnnotation> output_metadata_annotations_;
    std::shared_ptr<PipelineObserver> observer_;
    std::uint64_t next_event_sequence_ = 1;
    std::uint64_t next_buffer_sequence_ = 1;
    bool caching_enabled_ = true;
    std::size_t started_count_ = 0;
    std::stop_token stop_token_;
    std::map<const Element *, std::map<std::string, Buffer>> cached_outputs_;

    // Vector-clock provenance state (Phase 27). next_slot_ is the monotonic slot
    // allocator (0 reserved); element_base_ maps each slot-claiming element to its
    // base index; emit_counts_ tracks each element's production count.
    std::uint32_t next_slot_ = 1;
    std::map<const Element *, std::uint32_t> element_base_;
    std::map<const Element *, std::uint32_t> emit_counts_;
};

} // namespace leakflow
