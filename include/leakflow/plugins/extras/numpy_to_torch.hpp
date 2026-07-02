#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::extras {

inline constexpr auto numpy_to_torch_conversion_id = "numpy-to-torch";

class NumpyToTorch final : public Element {
public:
    explicit NumpyToTorch(std::string name = "numpytotorch0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::extras
