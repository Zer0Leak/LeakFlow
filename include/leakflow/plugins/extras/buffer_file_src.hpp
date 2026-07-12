#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/core/payload_codec.hpp"

#include <memory>
#include <optional>
#include <string>

namespace leakflow::plugins::extras {

// Reloads a Buffer previously written by BufferFileSink from a single HDF5 file:
// reads the envelope attributes (caps + metadata + payload type) and reconstructs the
// payload via the registered PayloadCodec through a BufferArchiveReader. Emits one
// buffer (one-run source). Its `src` pad declares ANY caps, so it links to any concrete
// sink; the emitted buffer carries the saved concrete caps.
class BufferFileSrc final : public Element {
public:
    explicit BufferFileSrc(std::shared_ptr<const PayloadCodecRegistry> codecs,
                           std::string name = "bufferfilesrc0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;

private:
    std::shared_ptr<const PayloadCodecRegistry> codecs_;
};

} // namespace leakflow::plugins::extras
