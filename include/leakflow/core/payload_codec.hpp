#pragma once

#include "leakflow/core/payload.hpp"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace leakflow {

// Serializes/deserializes one Payload type to/from a directory. Registered by the
// payload's type_name(); the generic BufferFileSink/BufferFileSrc drive it. Core
// holds only the callbacks -- the Torch/crypto-specific save/load lives in each
// plugin's codec, so leakflow_core stays domain-free (mirrors ElementFactoryRegistry).
struct PayloadCodec {
    // Write the payload's body into `dir` (which already exists). Free to create any
    // files it likes there (e.g. a torch-pickled payload.pt).
    std::function<void(const Payload& payload, const std::filesystem::path& dir)> save;
    // Rebuild the payload from the files a matching save() wrote into `dir`.
    std::function<std::shared_ptr<Payload>(const std::filesystem::path& dir)> load;
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
