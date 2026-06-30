#include "leakflow/core/buffer_queue.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

leakflow::Buffer numbered(int i)
{
    leakflow::Buffer buffer(leakflow::Caps("test/buf"));
    buffer.set_metadata("i", std::to_string(i));
    return buffer;
}

int value_of(const leakflow::Buffer& buffer)
{
    return std::stoi(buffer.metadata("i"));
}

// Producer thread pushes 0..N-1 through a SMALL bounded queue (so backpressure is
// exercised), then closes. Consumer pulls until EndOfStream. FIFO order must hold.
bool test_threaded_fifo_with_backpressure()
{
    constexpr int n = 200;
    leakflow::BufferQueue queue(4); // capacity 4 << 200 -> producer must block
    std::vector<int> received;
    std::atomic<std::size_t> max_size = 0;

    std::thread consumer([&] {
        std::stop_source never;
        while (true) {
            auto pull = queue.pull(never.get_token());
            if (pull.buffer) {
                received.push_back(value_of(*pull.buffer));
            } else if (pull.end_of_stream) {
                break;
            }
        }
    });

    std::thread producer([&] {
        std::stop_source never;
        for (int i = 0; i < n; ++i) {
            (void)queue.push(numbered(i), never.get_token());
            const auto s = queue.size();
            if (s > max_size) {
                max_size = s;
            }
        }
        queue.close();
    });

    producer.join();
    consumer.join();

    if (!expect(static_cast<int>(received.size()) == n, "queue did not deliver every buffer")) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (received[static_cast<std::size_t>(i)] != i) {
            std::cerr << "FIFO order broken at " << i << '\n';
            return false;
        }
    }
    if (!expect(max_size <= queue.capacity(), "queue exceeded its capacity (no backpressure)")) {
        return false;
    }
    return true;
}

bool test_close_then_drain_then_eos()
{
    leakflow::BufferQueue queue(8);
    std::stop_source never;
    (void)queue.push(numbered(1), never.get_token());
    (void)queue.push(numbered(2), never.get_token());
    queue.close();

    auto a = queue.pull(never.get_token());
    auto b = queue.pull(never.get_token());
    auto c = queue.pull(never.get_token());
    if (!expect(a.buffer && value_of(*a.buffer) == 1, "first drained buffer wrong")) {
        return false;
    }
    if (!expect(b.buffer && value_of(*b.buffer) == 2, "second drained buffer wrong")) {
        return false;
    }
    if (!expect(!c.buffer && c.end_of_stream, "closed+drained queue did not report EndOfStream")) {
        return false;
    }
    return true;
}

bool test_drop_policies()
{
    std::stop_source never;
    {
        leakflow::BufferQueue queue(2, leakflow::QueueDropPolicy::DropOldest);
        for (int i = 0; i < 5; ++i) {
            (void)queue.push(numbered(i), never.get_token());
        }
        queue.close();
        // capacity 2, DropOldest -> the two NEWEST remain: 3, 4.
        auto a = queue.pull(never.get_token());
        auto b = queue.pull(never.get_token());
        if (!expect(a.buffer && value_of(*a.buffer) == 3 && b.buffer && value_of(*b.buffer) == 4,
                "DropOldest did not keep the newest buffers")) {
            return false;
        }
    }
    {
        leakflow::BufferQueue queue(2, leakflow::QueueDropPolicy::DropNewest);
        for (int i = 0; i < 5; ++i) {
            (void)queue.push(numbered(i), never.get_token());
        }
        queue.close();
        // capacity 2, DropNewest -> the two OLDEST remain: 0, 1.
        auto a = queue.pull(never.get_token());
        auto b = queue.pull(never.get_token());
        if (!expect(a.buffer && value_of(*a.buffer) == 0 && b.buffer && value_of(*b.buffer) == 1,
                "DropNewest did not keep the oldest buffers")) {
            return false;
        }
    }
    return true;
}

// A consumer blocked on an empty queue must return NoData promptly when stop fires.
bool test_stop_unblocks_pull()
{
    using namespace std::chrono_literals;
    leakflow::BufferQueue queue(4);
    std::atomic<bool> got_nodata = false;
    std::atomic<bool> finished = false;

    std::jthread consumer([&](std::stop_token stop) {
        auto pull = queue.pull(stop); // blocks: empty and not closed
        if (!pull.buffer && !pull.end_of_stream) {
            got_nodata = true;
        }
        finished = true;
    });

    std::this_thread::sleep_for(50ms); // let it block
    consumer.request_stop();
    consumer.join();

    if (!expect(finished, "blocked pull did not return after stop")) {
        return false;
    }
    if (!expect(got_nodata, "stop did not unblock pull with NoData")) {
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!test_threaded_fifo_with_backpressure()) {
        return 1;
    }
    if (!test_close_then_drain_then_eos()) {
        return 1;
    }
    if (!test_drop_policies()) {
        return 1;
    }
    if (!test_stop_unblocks_pull()) {
        return 1;
    }
    return 0;
}
