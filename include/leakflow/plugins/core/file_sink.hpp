#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::core {

class FileSink final : public Element {
public:
    explicit FileSink(std::string name = "filesink0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;

    [[nodiscard]] bool received() const;
    [[nodiscard]] const std::optional<Buffer>& last_buffer() const;
    [[nodiscard]] const std::string& last_bytes() const;

private:
    bool received_ = false;
    std::optional<Buffer> last_buffer_;
    std::string last_bytes_;
};

} // namespace leakflow::plugins::core
