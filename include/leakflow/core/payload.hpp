#pragma once

#include "leakflow/core/summary_document.hpp"

#include <cstdint>
#include <string>

namespace leakflow {

class Payload {
public:
    virtual ~Payload() = default;

    [[nodiscard]] virtual std::string type_name() const = 0;
    virtual void describe(SummarySection& section, std::int64_t summary_level) const;
};

} // namespace leakflow
