#pragma once

#include "leakflow/core/buffer.hpp"

#include <stop_token>
#include <string>
#include <string_view>

namespace leakflow {

class Element;

// Narrow runtime service surface for future active elements.
//
// GStreamer spreads this responsibility across GstPad/GstTask/GstBus/state
// machinery. LeakFlow keeps it explicit: an element-owned task may push only from
// its own pads, report its own errors, and observe cooperative lifecycle state
// without receiving a mutable Pipeline pointer.
class RuntimeContext {
public:
    virtual ~RuntimeContext() = default;

    [[nodiscard]] virtual bool push(Element &element, std::string_view source_pad, Buffer buffer) = 0;
    virtual void end_of_stream(Element &element, std::string_view source_pad) = 0;
    virtual void report_error(Element &element, std::string message) = 0;
    virtual void safe_point(Element &element) = 0;

    [[nodiscard]] virtual std::stop_token stop_token() const = 0;
    [[nodiscard]] virtual bool stop_requested() const = 0;
    [[nodiscard]] virtual bool is_paused() const = 0;
    virtual void wait_if_paused() = 0;
};

} // namespace leakflow
