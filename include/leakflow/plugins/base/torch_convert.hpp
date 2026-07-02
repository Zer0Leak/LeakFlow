#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::base {

inline constexpr auto torch_convert_conversion_id = "torch-convert";

class TorchConvert final : public Element {
  public:
    explicit TorchConvert(std::string name = "torchconvert0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::base
