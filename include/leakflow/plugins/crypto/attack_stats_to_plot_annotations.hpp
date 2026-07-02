#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto attack_stats_to_plot_annotations_id =
    "attack-stats-to-plot-annotations";

class AttackStatsToPlotAnnotations final : public Element {
public:
    explicit AttackStatsToPlotAnnotations(std::string name = "attackstatsannotations0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::crypto
