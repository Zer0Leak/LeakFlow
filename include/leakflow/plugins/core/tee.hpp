#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>
#include <vector>

namespace leakflow::plugins::core {

class Tee final : public Element {
public:
    explicit Tee(std::string name = "tee0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] ElementOutputs process_pads(ElementInputs inputs) override;
    [[nodiscard]] std::vector<Buffer> fork(const Buffer& input) const;
};

} // namespace leakflow::plugins::core
