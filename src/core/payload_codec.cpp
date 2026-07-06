#include "leakflow/core/payload_codec.hpp"

#include <stdexcept>
#include <utility>

namespace leakflow {

void PayloadCodecRegistry::register_codec(std::string type_name, PayloadCodec codec)
{
    if (type_name.empty()) {
        throw std::invalid_argument("payload codec type_name must not be empty");
    }
    if (!codec.save || !codec.load) {
        throw std::invalid_argument("payload codec must provide both save and load");
    }
    codecs_.insert_or_assign(std::move(type_name), std::move(codec));
}

const PayloadCodec* PayloadCodecRegistry::find(std::string_view type_name) const
{
    const auto found = codecs_.find(type_name);
    return found == codecs_.end() ? nullptr : &found->second;
}

bool PayloadCodecRegistry::contains(std::string_view type_name) const
{
    return find(type_name) != nullptr;
}

} // namespace leakflow
