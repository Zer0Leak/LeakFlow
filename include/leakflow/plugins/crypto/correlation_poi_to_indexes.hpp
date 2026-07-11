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
// The `units` property (default [] = all) restricts/reorders the output to specific byte units
// (target byte indexes), so a single unit can be clustered -- pair it with the same subset on
// the truth side, e.g. AesLeakage(byte_indexes=[0]), to keep the U axes aligned.
class CorrelationPoiToIndexes final : public Element {
public:
    explicit CorrelationPoiToIndexes(std::string name = "poiindexes0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::crypto
