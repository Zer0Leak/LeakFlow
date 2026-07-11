#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::ml {

// Gathers feature columns by index (Transform/Feature/Select). Generic: given a feature tensor
// on `features` and an integer index tensor on `indexes`, emits the selected columns on
// `selected`. Motivating use: truncate shared traces `[T, N]` to per-unit PoI columns
// `[U, N_sel]` -> `[U, T, N_sel]` for one GMM per unit. Pure function of its inputs, so
// can_replay() is true.
//
// Shapes: features `[T, N]` or `[U, T, N]`; indexes `[N_sel]` (shared) or `[U, N_sel]`
// (per unit). 1-D indexes select the same columns for every unit; 2-D indexes select per unit
// and broadcast a shared `[T, N]` feature matrix across the U axis.
class FeatureSelect final : public Element {
public:
    explicit FeatureSelect(std::string name = "featureselect0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] std::optional<Buffer> process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::ml
