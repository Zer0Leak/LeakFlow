#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto cpa_attack_stats_id = "cpa-attack-stats";

class CpaAttackStats final : public Element {
public:
    explicit CpaAttackStats(std::string name = "cpaattackstats0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::crypto
