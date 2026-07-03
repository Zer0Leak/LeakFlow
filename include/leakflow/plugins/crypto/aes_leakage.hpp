#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/crypto/aes.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto aes_leakage_model_id = "aes-first-round";
inline constexpr auto aes_leakage_channel_hw_m =
    leakflow::crypto::aes::first_round_leakage_channel_hw_m;
inline constexpr auto aes_leakage_channel_hw_m_xor_k =
    leakflow::crypto::aes::first_round_leakage_channel_hw_m_xor_k;
inline constexpr auto aes_leakage_channel_hw_y =
    leakflow::crypto::aes::first_round_leakage_channel_hw_y;
inline constexpr auto aes_leakage_channel_y_bits =
    leakflow::crypto::aes::first_round_leakage_channel_y_bits;

class AesLeakage final : public Element {
public:
  explicit AesLeakage(std::string name = "aesleakage0");

  [[nodiscard]] static ElementDescriptor descriptor();
  [[nodiscard]] std::optional<Buffer>
  process(std::optional<Buffer> input) override;
  [[nodiscard]] std::optional<Buffer>
  process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::crypto
