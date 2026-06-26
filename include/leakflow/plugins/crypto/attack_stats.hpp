#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto attack_stats_id = "attack-stats";

class AttackStats final : public Element {
public:
    explicit AttackStats(std::string name = "attackstats0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::crypto
