#pragma once

#include "leakflow/core/payload.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

// The archive types are defined in leakflow_base (Torch-aware). Core only names
// them in the codec signatures below, so a forward declaration keeps
// leakflow_core free of Torch and of the concrete storage backend.
namespace leakflow::base {
class BufferArchiveWriter;
class BufferArchiveReader;
} // namespace leakflow::base

namespace leakflow {

// Serializes/deserializes one Payload type through a storage-neutral archive.
// Registered by the payload's type_name(); the BufferFileSink/BufferFileSrc
// elements drive it. Core holds only the callbacks -- the Torch/crypto-specific
// mapping lives in each plugin's codec, and the concrete backend (HDF5 today,
// Zarr later) is a BufferArchiveWriter/Reader implementation in a higher layer,
// so leakflow_core stays domain-free (mirrors ElementFactoryRegistry).
struct PayloadCodec {
    // Write the payload's body into the archive as named tensors/scalars.
    std::function<void(const Payload& payload, base::BufferArchiveWriter& archive)> save;
    // Rebuild the payload from the entries a matching save() wrote.
    std::function<std::shared_ptr<Payload>(const base::BufferArchiveReader& archive)> load;
};

class PayloadCodecRegistry {
public:
    // type_name is the payload's Payload::type_name() (its stable serialization id).
    void register_codec(std::string type_name, PayloadCodec codec);
    [[nodiscard]] const PayloadCodec* find(std::string_view type_name) const;
    [[nodiscard]] bool contains(std::string_view type_name) const;

private:
    std::map<std::string, PayloadCodec, std::less<>> codecs_;
};

} // namespace leakflow
