#include "leakflow/plugins/base/app_src.hpp"

#include "leakflow/base/numeric_caps.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow::plugins::base {

ElementDescriptor AppSrc::descriptor()
{
    return {
        .type_name = "AppSrc",
        .long_name = "Application Source",
        .rank = ElementRank::Primary,
        .klass = "Source/App/Torch",
        .purpose = "emit application-pushed Torch tensor frames (one frame = aligned buffers on src_0..src_N)",
        .pad_templates =
            {
                Pad("src_%u", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type),
                    PadPresence::OnRequest),
            },
        .keywords = {"app", "appsrc", "push", "source", "live", "base"},
        // provenance_slots defaults to 1: AppSrc claims one slot and every buffer of
        // a frame is emitted in a single firing, so the executor stamps them with the
        // same clock value (common-ancestor injection) and downstream joins pair them.
        .live_source = true,
    };
}

AppSrc::AppSrc(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

void AppSrc::set_frame_producer(FrameProducer producer)
{
    producer_ = std::move(producer);
}

ElementOutputs AppSrc::frame_to_outputs(std::vector<Buffer> frame)
{
    ElementOutputs outputs;
    for (std::size_t index = 0; index < frame.size(); ++index) {
        outputs.emplace("src_" + std::to_string(index), std::move(frame[index]));
    }
    return outputs;
}

AppSrc::ProgressReport AppSrc::make_progress_report()
{
    return [this](double fraction, std::string_view message, std::uint64_t index, std::uint64_t total) {
        report_progress(fraction, std::string(message), index, total);
    };
}

void AppSrc::start()
{
    // Pull mode: rewind to frame 0 and prefetch it, so a Stop -> Start cycle
    // re-streams from the beginning and at_end_of_stream() is exact from the first
    // pump check (an empty producer ends the run before any sweep).
    if (producer_) {
        cursor_ = 0;
        pending_ = producer_(cursor_++, make_progress_report());
    }
}

void AppSrc::push_frame(std::vector<Buffer> frame)
{
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            throw std::logic_error("AppSrc::push_frame called after end_of_stream");
        }
        frames_.push_back(std::move(frame));
    }
    not_empty_.notify_one();
}

void AppSrc::end_of_stream()
{
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    not_empty_.notify_all();
}

std::optional<Buffer> AppSrc::process(std::optional<Buffer>)
{
    throw std::invalid_argument("AppSrc emits named output pads (src_0, src_1, ...); use process_pads");
}

ElementOutputs AppSrc::process_pads(ElementInputs)
{
    if (producer_) {
        // Pull mode: emit the prefetched frame and prefetch the next. The prefetch
        // means at_end_of_stream() already reports true when the last frame was
        // emitted, so the pump never enters an empty sweep.
        if (!pending_) {
            return {};
        }
        auto frame = std::move(*pending_);
        pending_ = producer_(cursor_++, make_progress_report());
        return frame_to_outputs(std::move(frame));
    }

    std::unique_lock<std::mutex> lock(mutex_);
    // Wait for a frame or for the stream to close. The stop_token overload unblocks
    // on cooperative stop (Ctrl+C / window close) so the pump can unwind promptly.
    not_empty_.wait(lock, stop_token(), [this] { return !frames_.empty() || closed_; });
    if (frames_.empty()) {
        // Closed and drained: end of stream. Returns no outputs; at_end_of_stream()
        // is already true, so the pump stops before entering another sweep.
        return {};
    }

    auto frame = std::move(frames_.front());
    frames_.pop_front();
    lock.unlock();

    auto outputs = frame_to_outputs(std::move(frame));
    auto record = make_log_record(log::LogLevel::Debug, "element", "emitted application frame");
    record.fields.emplace("frame_pads", std::to_string(outputs.size()));
    leakflow::log::write(std::move(record));

    return outputs;
}

bool AppSrc::at_end_of_stream() const
{
    if (producer_) {
        return !pending_.has_value();
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    return closed_ && frames_.empty();
}

} // namespace leakflow::plugins::base
