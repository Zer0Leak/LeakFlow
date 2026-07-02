#include "leakflow/core/payload.hpp"

namespace leakflow {

void Payload::describe(SummarySection& section, std::int64_t) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
}

} // namespace leakflow
