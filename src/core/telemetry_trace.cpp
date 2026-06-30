#include "leakflow/core/telemetry_trace.hpp"

#include <cstddef>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

namespace leakflow {

namespace {

[[nodiscard]] std::uint64_t current_thread_id()
{
    return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

void append_json_escaped(std::string &out, std::string_view text)
{
    for (const char character : text) {
        switch (character) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\r':
            out += "\\r";
            break;
        default:
            out += character;
            break;
        }
    }
}

} // namespace

void TelemetryTraceSink::add_complete(std::string name, std::string category,
    std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    const auto begin_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(begin - epoch_).count();
    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    const std::lock_guard<std::mutex> lock(mutex_);
    spans_.push_back(TelemetryTraceSpan{
        .name = std::move(name),
        .category = std::move(category),
        .thread_id = current_thread_id(),
        .timestamp_ns = begin_ns < 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(begin_ns),
        .duration_ns = duration_ns < 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(duration_ns),
    });
}

bool TelemetryTraceSink::empty() const
{
    const std::lock_guard<std::mutex> lock(mutex_);
    return spans_.empty();
}

std::size_t TelemetryTraceSink::span_count() const
{
    const std::lock_guard<std::mutex> lock(mutex_);
    return spans_.size();
}

std::string TelemetryTraceSink::to_chrome_json() const
{
    const std::lock_guard<std::mutex> lock(mutex_);

    // One track (tid) per element category, not per OS thread. An element processes
    // one buffer at a time, so its complete (X) slices are always sequential and its
    // op scopes nest strictly within them -- no overlapping slices on a track, which
    // is what Perfetto rejects. This also holds across --graph re-runs and is the
    // natural per-element timeline view. The real OS thread is kept in args.
    std::map<std::string, std::size_t> track_ids;
    for (const auto &span : spans_) {
        track_ids.emplace(span.category, track_ids.size() + 1);
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "[\n";

    // Metadata events label each element track and the process.
    out << "  {\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"LeakFlow\"}}";
    for (const auto &[category, tid] : track_ids) {
        std::string label;
        append_json_escaped(label, category);
        out << ",\n  {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << tid
            << ",\"args\":{\"name\":\"" << label << "\"}}";
    }

    for (const auto &span : spans_) {
        std::string name;
        append_json_escaped(name, span.name);
        std::string category;
        append_json_escaped(category, span.category);

        out << ",\n  {\"name\":\"" << name << "\",\"cat\":\"" << category
            << "\",\"ph\":\"X\",\"pid\":1,\"tid\":" << track_ids.at(span.category)
            << ",\"ts\":" << static_cast<double>(span.timestamp_ns) / 1000.0
            << ",\"dur\":" << static_cast<double>(span.duration_ns) / 1000.0
            << ",\"args\":{\"os_thread\":" << span.thread_id << "}}";
    }
    out << "\n]\n";
    return out.str();
}

} // namespace leakflow
