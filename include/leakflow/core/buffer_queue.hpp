#pragma once

#include "leakflow/core/buffer.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stop_token>

namespace leakflow {

// Drop policy for a bounded queue (live phase). Block backpressures the producer;
// DropOldest/DropNewest never block (evict to make room) -- needed for hardware
// captures that cannot be back-pressured. Aligns with QueueEpochPolicy's
// Block/DropOldest/DropNewest values.
enum class QueueDropPolicy { Block, DropOldest, DropNewest };

// Thread-safe bounded FIFO of Buffers (live phase, step 1). The runtime backing of
// the Queue element and the hand-off between two segment threads. This is the one
// place the concurrency lives; elements above it stay thread-unaware. Implements
// the 3-state pull (Data / NoData / EndOfStream) and cooperative stop from
// docs/design/dataflow_sync_model.md S11.8. Not yet wired into the executor.
class BufferQueue {
public:
    struct Pull {
        std::optional<Buffer> buffer; // present iff Data
        bool end_of_stream = false;   // true iff producer closed AND queue drained
        std::size_t size_after = 0;   // queue depth after the pull decision
        // buffer && !end_of_stream  -> Data
        // !buffer && !end_of_stream -> NoData (stop requested / transient)
        // !buffer && end_of_stream  -> EndOfStream
    };

    struct Push {
        bool accepted = false;      // false only when stop/close prevented progress
        bool dropped = false;       // true when a drop policy discarded a buffer
        std::size_t size_after = 0; // queue depth after the push decision
    };

    explicit BufferQueue(std::size_t capacity, QueueDropPolicy drop_policy = QueueDropPolicy::Block);

    // Producer side. Block: parks until space, stop, or close. Drop policies never
    // block. Returns false only when stop/close prevented accepting the buffer.
    bool push(Buffer buffer, std::stop_token stop);

    // Same producer operation with exact telemetry collected while holding the
    // queue mutex. The bool-only push() API remains for simple callers.
    [[nodiscard]] Push push_with_status(Buffer buffer, std::stop_token stop);

    // Consumer side. Blocks until a buffer is available, stop is requested, or the
    // queue is closed-and-drained. See Pull for the 3-state result.
    [[nodiscard]] Pull pull(std::stop_token stop);

    // Non-blocking consumer side: returns immediately. Data if a buffer is ready,
    // EndOfStream if closed and drained, NoData otherwise. Used by the Latest join
    // mode to sample the newest buffer without waiting.
    [[nodiscard]] Pull try_pull();

    // Producer signals end-of-stream. Drained consumers then see EndOfStream;
    // blocked producers/consumers wake.
    void close();

    [[nodiscard]] bool is_closed() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable_any not_full_;
    std::condition_variable_any not_empty_;
    std::deque<Buffer> buffers_;
    std::size_t capacity_;
    QueueDropPolicy drop_policy_;
    bool closed_ = false;
};

} // namespace leakflow
