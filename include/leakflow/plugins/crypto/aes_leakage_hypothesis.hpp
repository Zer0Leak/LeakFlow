#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"
#include "leakflow/plugins/crypto/aes_leakage.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto aes_leakage_hypothesis_id =
    "aes-first-round-leakage-hypothesis";

class AesLeakageHypothesis final : public Element {
public:
  explicit AesLeakageHypothesis(std::string name = "aesleakagehypothesis0");

  [[nodiscard]] static ElementDescriptor descriptor();
  [[nodiscard]] std::optional<Buffer>
  process(std::optional<Buffer> input) override;
  [[nodiscard]] std::optional<Buffer>
  process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::crypto
