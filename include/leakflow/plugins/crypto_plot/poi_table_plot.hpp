#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <memory>
#include <optional>
#include <string>

namespace leakflow::plot {
class PoiTableView;
}

namespace leakflow::plugins::crypto_plot {

// Renders a two-row PoI comparison table (Plot/Table/PoI): a `reference` PoI payload (e.g.
// profiling PoIs) against a `current` one (e.g. re-scored on attack traces), per (unit, channel)
// with unit and channel sliders. Columns are the PoI sample indexes when the payload carries
// them (result last axis = 2 -> (index, score)), or 1..N ordinals for score-only payloads
// (last axis = 1). Both inputs are optional; a missing side shows "-".
class PoiTablePlot final : public Element {
public:
    explicit PoiTablePlot(std::string name = "poitable0");
    PoiTablePlot(std::shared_ptr<leakflow::plot::PoiTableView> view, std::string name = "poitable0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;

    [[nodiscard]] std::shared_ptr<leakflow::plot::PoiTableView> poi_table_view() const;

private:
    std::shared_ptr<leakflow::plot::PoiTableView> view_;
};

} // namespace leakflow::plugins::crypto_plot
