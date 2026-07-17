#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

// Converts a CorrelationPoiPayload into a plain int64 index tensor `[U, N_sel]`
// (Convert/SCA/PoiIndexes): per target byte, the selected PoI sample indexes across all
// channels are concatenated (N_sel = channels * top_k). This bridges PoiSelect to the generic
// FeatureSelect element, which truncates traces to those columns for one GMM per byte.
//
// This is a pure format converter: it flattens whatever units/channels the payload carries, in
// payload order, and does not itself subset them. Semantic-axis selection (which byte units,
// which leakage channels) belongs upstream on PoiSelect (its `units` / `channels` properties),
// so pair a single-unit PoiSelect(units=[...]) with the matching truth side, e.g.
// AesLeakage(units=[...]), to keep the U axes aligned.
class CorrelationPoiToIndexes final : public Element {
public:
    explicit CorrelationPoiToIndexes(std::string name = "poiindexes0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::crypto
