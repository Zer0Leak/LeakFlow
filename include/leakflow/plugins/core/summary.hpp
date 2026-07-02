#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::core {

class Summary final : public Element {
public:
    explicit Summary(std::string name = "summary0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;

    [[nodiscard]] const std::string& last_summary() const;
    [[nodiscard]] const std::string& last_plain_summary() const;

private:
    std::string last_summary_;
    std::string last_plain_summary_;
};

} // namespace leakflow::plugins::core
