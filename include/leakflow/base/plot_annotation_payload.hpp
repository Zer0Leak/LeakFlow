#pragma once

#include "leakflow/core/caps.hpp"
#include "leakflow/core/payload.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace leakflow::base {

inline constexpr auto plot_annotation_caps_type = "leakflow/plot-annotations";

using PlotAnnotationFields = std::vector<std::pair<std::string, std::string>>;

struct PlotAnnotation {
    std::int64_t sample_index = 0;
    std::optional<double> value;
    std::optional<double> norm_value;
    PlotAnnotationFields fields;
    std::string label;
    std::string text;
    std::string kind;
    std::optional<std::int64_t> target_index;
    // Generic marker-shape hint for renderers: "circle" (default when empty),
    // "square", or "x". Producers use it to encode a categorical status (for
    // example AttackStats success: square=true, x=false, circle=unknown).
    std::string marker;
};

class PlotAnnotationPayload final : public Payload {
public:
    explicit PlotAnnotationPayload(std::vector<PlotAnnotation> annotations);

    [[nodiscard]] std::string type_name() const override;
    [[nodiscard]] std::string layout() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    [[nodiscard]] const std::vector<PlotAnnotation>& annotations() const;
    [[nodiscard]] const PlotAnnotation& annotation(std::size_t index) const;
    [[nodiscard]] std::size_t annotation_count() const;
    [[nodiscard]] Caps caps() const;

private:
    std::vector<PlotAnnotation> annotations_;
};

} // namespace leakflow::base
