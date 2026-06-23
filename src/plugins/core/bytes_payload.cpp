#include "leakflow/plugins/core/bytes_payload.hpp"

#include <utility>

namespace leakflow::plugins::core {

BytesPayload::BytesPayload(std::string bytes)
    : bytes_(std::move(bytes))
{
}

std::string BytesPayload::type_name() const
{
    return "leakflow/bytes";
}

const std::string& BytesPayload::bytes() const
{
    return bytes_;
}

std::string& BytesPayload::bytes()
{
    return bytes_;
}

} // namespace leakflow::plugins::core

