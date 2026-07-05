#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace leakflow::plugins::base {

// AppSrc: an application-fed live source ("appsrc"). Instead of reading a file or
// device, the application pushes buffers into it and the pipeline consumes them as
// if from a real source. It is the general seam for feeding external data (folders
// of .pt files now, hardware/generated later) into a pipeline without a new source
// element per dataset layout.
//
// One push is one *frame* = an aligned set of buffers routed to src_0..src_{n-1}.
// All buffers of a frame are emitted in a single firing, so the executor stamps
// them with one shared vector clock (common-ancestor injection, like Sync) and
// every downstream join pairs them per frame with the default barrier.
//
// It declares itself live, so an accumulating downstream element (e.g.
// PearsonPoiFinder in auto/incremental mode) folds each frame into running state
// and never resets between frames -- only at start().
//
// v1 usage is synchronous drained: push every frame and call end_of_stream()
// before run(); process_pads() then never has to report NoData, so the single-
// threaded pump stays free of empty sweeps. Concurrent bounded push (a producer
// thread + the threaded/Queue path) is a later refinement.
class AppSrc final : public Element {
public:
    explicit AppSrc(std::string name = "appsrc0");

    [[nodiscard]] static ElementDescriptor descriptor();

    // Returns frame `index` (aligned buffers for src_0..src_N), or nullopt for
    // end-of-stream. Called on the pipeline worker thread. AppSrc owns the index
    // (0-based, advancing per pump step) and resets it to 0 in start(), so a
    // Stop -> Start cycle re-streams from the beginning.
    using FrameProducer = std::function<std::optional<std::vector<Buffer>>(std::size_t index)>;

    // Pull / lazy mode (preferred for streaming many large frames): the source asks
    // the application for the next frame when it needs one, instead of holding them
    // all up front. It prefetches one frame ahead so end-of-stream is known before a
    // sweep (no empty sweep) and only ~one frame is in memory at a time -- so the
    // graph window opens immediately and memory stays bounded. Set before run().
    void set_frame_producer(FrameProducer producer);

    // Push mode (thread-safe): the application enqueues frames; buffer i routes to
    // output pad "src_i". Throws after end_of_stream(). v1 push usage pre-fills every
    // frame and closes before run(). Ignored while a frame producer is set.
    void push_frame(std::vector<Buffer> frame);

    // Push mode: signal that no more frames will be pushed. Once the queue drains,
    // at_end_of_stream() reports true and the pump stops.
    void end_of_stream();

    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] ElementOutputs process_pads(ElementInputs inputs) override;
    [[nodiscard]] bool at_end_of_stream() const override;

private:
    [[nodiscard]] static ElementOutputs frame_to_outputs(std::vector<Buffer> frame);

    // Pull mode: worker-thread only (start() + process_pads() + at_end_of_stream()
    // all run on the pump thread), so these need no lock.
    FrameProducer producer_;
    std::optional<std::vector<Buffer>> pending_;
    std::size_t cursor_ = 0;

    // Push mode: cross-thread queue.
    mutable std::mutex mutex_;
    std::condition_variable_any not_empty_;
    std::deque<std::vector<Buffer>> frames_;
    bool closed_ = false;
};

} // namespace leakflow::plugins::base
