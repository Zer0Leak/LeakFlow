#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::base {

class TorchFileSink final : public Element {
public:
    explicit TorchFileSink(std::string name = "torchfilesink0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;

    [[nodiscard]] bool received() const;
    [[nodiscard]] const std::optional<Buffer>& last_buffer() const;

private:
    bool received_ = false;
    std::optional<Buffer> last_buffer_;
};

} // namespace leakflow::plugins::base
