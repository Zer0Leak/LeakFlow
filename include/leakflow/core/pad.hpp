#pragma once

#include "leakflow/core/caps.hpp"

#include <string>

namespace leakflow {

enum class PadDirection {
    Input,
    Output,
};

enum class PadPresence {
    Required,
    Optional,
    OnRequest,
};

class Pad {
public:
    Pad(std::string name,
        PadDirection direction,
        Caps caps,
        PadPresence presence = PadPresence::Required);

    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] PadDirection direction() const;
    [[nodiscard]] const Caps& caps() const;
    [[nodiscard]] PadPresence presence() const;
    [[nodiscard]] bool is_required() const;

private:
    std::string name_;
    PadDirection direction_;
    Caps caps_;
    PadPresence presence_;
};

} // namespace leakflow
