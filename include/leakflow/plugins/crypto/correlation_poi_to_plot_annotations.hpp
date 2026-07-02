#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

inline constexpr auto correlation_poi_to_plot_annotations_id = "correlation-poi-to-plot-annotations";

class CorrelationPoiToPlotAnnotations final : public Element {
public:
    explicit CorrelationPoiToPlotAnnotations(std::string name = "correlationpoitoplotannotations0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::crypto
