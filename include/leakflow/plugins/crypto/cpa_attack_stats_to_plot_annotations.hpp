#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto cpa_attack_stats_to_plot_annotations_id =
    "cpa-attack-stats-to-plot-annotations";

class CpaAttackStatsToPlotAnnotations final : public Element {
public:
    explicit CpaAttackStatsToPlotAnnotations(std::string name = "cpaattackstatsannotations0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::crypto
