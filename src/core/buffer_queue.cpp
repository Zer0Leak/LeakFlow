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
    std::unique_lock<std::mutex> lock(mutex_);

    if (drop_policy_ == QueueDropPolicy::Block) {
        // Park until there is room, the queue closes, or stop is requested.
        const bool ready = not_full_.wait(lock, stop, [this] { return buffers_.size() < capacity_ || closed_; });
        if (!ready || closed_) {
            return false; // stop requested or closed: not accepted
        }
        buffers_.push_back(std::move(buffer));
        not_empty_.notify_one();
        return true;
    }

    if (closed_) {
        return false;
    }
    if (buffers_.size() >= capacity_) {
        if (drop_policy_ == QueueDropPolicy::DropOldest) {
            buffers_.pop_front();
        } else { // DropNewest: drop the incoming buffer
            return true;
        }
    }
    buffers_.push_back(std::move(buffer));
    not_empty_.notify_one();
    return true;
}

BufferQueue::Pull BufferQueue::pull(std::stop_token stop)
{
    std::unique_lock<std::mutex> lock(mutex_);
    (void)not_empty_.wait(lock, stop, [this] { return !buffers_.empty() || closed_; });

    if (!buffers_.empty()) {
        Buffer buffer = std::move(buffers_.front());
        buffers_.pop_front();
        not_full_.notify_one();
        return Pull{std::move(buffer), false}; // Data
    }
    if (closed_) {
        return Pull{std::nullopt, true}; // EndOfStream (closed and drained)
    }
    return Pull{std::nullopt, false}; // NoData (stop requested / transient)
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
