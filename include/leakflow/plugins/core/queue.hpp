#pragma once

#include "leakflow/core/active_element.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/core/buffer_queue.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/core/thread_boundary_runtime.hpp"

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

namespace leakflow::plugins::core {

class Queue final : public Element, public ThreadBoundaryRuntime, public ActiveElement {
public:
    explicit Queue(std::string name = "queue0");

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] bool can_replay() const override;
    [[nodiscard]] std::size_t depth() const;

    void prepare_thread_boundary_runtime(std::mutex *property_mutex) override;
    void clear_thread_boundary_runtime() noexcept override;
    [[nodiscard]] bool boundary_push(Buffer buffer, std::stop_token stop) override;
    [[nodiscard]] BufferQueue::Pull boundary_pull(std::stop_token stop) override;
    [[nodiscard]] BufferQueue::Pull boundary_try_pull() override;
    void boundary_close() override;
    [[nodiscard]] std::vector<TelemetrySnapshot> telemetry_snapshot() const override;
    [[nodiscard]] std::vector<TelemetrySnapshot> boundary_runtime_telemetry() override;

    void start_active(RuntimeContext &context) override;
    void wait_active() override;
    void stop_active() noexcept override;

private:
    struct TelemetryCounters {
        std::size_t size = 0;
        std::size_t peak_size = 0;
        std::size_t received = 0;
        std::size_t emitted = 0;
        std::size_t dropped = 0;
    };

    void reset_telemetry();
    void record_telemetry_size(std::size_t size);
    void record_telemetry_received(std::size_t size_after, bool dropped);
    void record_telemetry_emitted(std::size_t size_after);
    [[nodiscard]] TelemetryCounters telemetry_counters() const;
    [[nodiscard]] BufferQueue &runtime_queue();

    std::deque<Buffer> buffers_;
    std::unique_ptr<BufferQueue> runtime_queue_;
    RuntimeTelemetrySizeGauge telemetry_size_;
    RuntimeTelemetrySizePeak telemetry_peak_size_;
    RuntimeTelemetrySizeCounter telemetry_received_;
    RuntimeTelemetrySizeCounter telemetry_emitted_;
    RuntimeTelemetrySizeCounter telemetry_dropped_;
    std::jthread active_worker_;
};

} // namespace leakflow::plugins::core
