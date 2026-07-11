#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

// Re-scores existing PoIs on new traces (Analyze/SCA/PoiCorrelation): keeps the PoI sample
// positions from an input CorrelationPoiPayload but recomputes each PoI's Pearson correlation
// against a fresh leakage tensor on new (e.g. attack) traces. Emits a CorrelationPoiPayload with
// the same positions and the *new* correlation values -- so it plugs straight into
// CorrelationPoiToPlotAnnotations and overlays on the original PoIs, showing how well each
// profiling PoI still leaks on the new set.
//
// Inputs: `poi` (the profiling CorrelationPoiPayload), `traces` (new traces [T, S]), `targets`
// (new AesLeakage [B, N, C], byte/channel order matching the PoI groups). Output: `poi`.
class PoiCorrelation final : public Element {
public:
    explicit PoiCorrelation(std::string name = "poicorr0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::crypto
