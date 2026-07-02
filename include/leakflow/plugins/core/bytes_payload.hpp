#pragma once

#include "leakflow/core/payload.hpp"

#include <string>

namespace leakflow::plugins::core {

class BytesPayload final : public Payload {
public:
    explicit BytesPayload(std::string bytes = {});

    [[nodiscard]] std::string type_name() const override;
    [[nodiscard]] const std::string& bytes() const;
    [[nodiscard]] std::string& bytes();

private:
    std::string bytes_;
};

} // namespace leakflow::plugins::core

