#include "leakflow/core/pad.hpp"

#include <utility>

namespace leakflow {

Pad::Pad(std::string name, PadDirection direction, Caps caps, PadPresence presence)
    : name_(std::move(name))
    , direction_(direction)
    , caps_(std::move(caps))
    , presence_(presence)
{
}

const std::string& Pad::name() const
{
    return name_;
}

PadDirection Pad::direction() const
{
    return direction_;
}

const Caps& Pad::caps() const
{
    return caps_;
}

PadPresence Pad::presence() const
{
    return presence_;
}

bool Pad::is_required() const
{
    return presence_ == PadPresence::Required;
}

} // namespace leakflow
