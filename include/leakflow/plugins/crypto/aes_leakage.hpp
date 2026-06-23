#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto aes_leakage_model_id = "aes-first-round";
inline constexpr auto aes_leakage_channel_hw_m = "HW(m)";
inline constexpr auto aes_leakage_channel_hw_m_xor_k = "HW(m_xor_k)";
inline constexpr auto aes_leakage_channel_hw_y = "HW(y)";

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
