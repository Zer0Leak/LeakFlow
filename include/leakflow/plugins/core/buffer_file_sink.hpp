#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/core/payload_codec.hpp"

#include <memory>
#include <optional>
#include <string>

namespace leakflow::plugins::core {

// Persists a whole Buffer (caps + metadata + payload) to a directory: a
// human-readable `manifest.txt` envelope plus the payload body written by the
// PayloadCodec registered for the payload's type_name(). Generic -- it never knows
// the concrete payload type; BufferFileSrc reloads it. See buffer_file_manifest.hpp.
class BufferFileSink final : public Element {
public:
    explicit BufferFileSink(std::shared_ptr<const PayloadCodecRegistry> codecs,
                            std::string name = "bufferfilesink0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;

private:
    std::shared_ptr<const PayloadCodecRegistry> codecs_;
};

} // namespace leakflow::plugins::core
