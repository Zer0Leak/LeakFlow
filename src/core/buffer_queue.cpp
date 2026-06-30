#include "leakflow/core/buffer_queue.hpp"

#include <utility>

namespace leakflow {

BufferQueue::BufferQueue(std::size_t capacity, QueueDropPolicy drop_policy)
    : capacity_(capacity == 0 ? 1 : capacity)
    , drop_policy_(drop_policy)
{
}

bool BufferQueue::push(Buffer buffer, std::stop_token stop)
{
    return push_with_status(std::move(buffer), stop).accepted;
}

BufferQueue::Push BufferQueue::push_with_status(Buffer buffer, std::stop_token stop)
{
    std::unique_lock<std::mutex> lock(mutex_);

    if (drop_policy_ == QueueDropPolicy::Block) {
        // Park until there is room, the queue closes, or stop is requested.
        const bool ready = not_full_.wait(lock, stop, [this] { return buffers_.size() < capacity_ || closed_; });
        if (!ready || closed_) {
            return Push{false, false, buffers_.size()}; // stop requested or closed: not accepted
        }
        buffers_.push_back(std::move(buffer));
        const auto size_after = buffers_.size();
        not_empty_.notify_one();
        return Push{true, false, size_after};
    }

    if (closed_) {
        return Push{false, false, buffers_.size()};
    }
    bool dropped = false;
    if (buffers_.size() >= capacity_) {
        if (drop_policy_ == QueueDropPolicy::DropOldest) {
            buffers_.pop_front();
            dropped = true;
        } else { // DropNewest: drop the incoming buffer
            return Push{true, true, buffers_.size()};
        }
    }
    buffers_.push_back(std::move(buffer));
    const auto size_after = buffers_.size();
    not_empty_.notify_one();
    return Push{true, dropped, size_after};
}

BufferQueue::Pull BufferQueue::pull(std::stop_token stop)
{
    std::unique_lock<std::mutex> lock(mutex_);
    (void)not_empty_.wait(lock, stop, [this] { return !buffers_.empty() || closed_; });

    if (!buffers_.empty()) {
        Buffer buffer = std::move(buffers_.front());
        buffers_.pop_front();
        const auto size_after = buffers_.size();
        not_full_.notify_one();
        return Pull{std::move(buffer), false, size_after}; // Data
    }
    if (closed_) {
        return Pull{std::nullopt, true, buffers_.size()}; // EndOfStream (closed and drained)
    }
    return Pull{std::nullopt, false, buffers_.size()}; // NoData (stop requested / transient)
}

BufferQueue::Pull BufferQueue::try_pull()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!buffers_.empty()) {
        Buffer buffer = std::move(buffers_.front());
        buffers_.pop_front();
        const auto size_after = buffers_.size();
        not_full_.notify_one();
        return Pull{std::move(buffer), false, size_after}; // Data
    }
    if (closed_) {
        return Pull{std::nullopt, true, buffers_.size()}; // EndOfStream
    }
    return Pull{std::nullopt, false, buffers_.size()}; // NoData (empty, not closed)
}

void BufferQueue::close()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
}

bool BufferQueue::is_closed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
}

std::size_t BufferQueue::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return buffers_.size();
}

std::size_t BufferQueue::capacity() const
{
    return capacity_;
}

} // namespace leakflow
