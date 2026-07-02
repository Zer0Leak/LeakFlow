#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::base {

class TorchFileSrc final : public Element {
public:
    explicit TorchFileSrc(std::string name = "torchfilesrc0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::base
