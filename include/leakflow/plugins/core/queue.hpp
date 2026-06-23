#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <string>

namespace leakflow::plugins::core {

class Queue final : public Element {
public:
    explicit Queue(std::string name = "queue0");

    [[nodiscard]] static ElementDescriptor descriptor();
    void start() override;
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] bool can_replay() const override;
    [[nodiscard]] std::size_t depth() const;

private:
    std::deque<Buffer> buffers_;
};

} // namespace leakflow::plugins::core
