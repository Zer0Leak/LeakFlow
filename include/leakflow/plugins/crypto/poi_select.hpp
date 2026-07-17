#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

// Kept for the annotation bridge and reports: the PoI selection method id. The
// CorrelationPoiToPlotAnnotations converter checks payload.poi.method against this.
inline constexpr auto pearson_poi_method_id = "pearson-correlation";

// Selects the top-k points of interest per (unit, channel) from a CorrelationPayload
// and emits a CorrelationPoiPayload. It also owns semantic-axis selection: the `units`
// property keeps/reorders byte units and the `channels` property keeps/reorders leakage
// channels (by name), so a single (unit, channel) slice can be carried downstream without a
// separate converter re-subsetting it. This is the *stateless* half of the old
// PearsonPoiFinder: its output is a pure function of (correlation, top_k, rank_by, units,
// channels), so it is replayable (can_replay() default true) -- changing any of them in Idle
// re-selects from the cached correlation through the normal partial-rerun path, no
// re-accumulation. The accumulation lives upstream in PearsonCorrelator.
class PoiSelect final : public Element {
public:
    explicit PoiSelect(std::string name = "poiselect0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::crypto
