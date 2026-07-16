#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::ml {

// Compare arbitrary predicted cluster IDs with exact vector-valued semantic
// truth. This is the structured evaluator; the legacy ClusteringStats matrix
// adapter remains unchanged.
class ClusteringEvaluate final : public Element {
public:
  explicit ClusteringEvaluate(std::string name = "clusteringevaluate0");

  [[nodiscard]] static ElementDescriptor descriptor();
  [[nodiscard]] std::optional<Buffer>
  process(std::optional<Buffer> input) override;
  [[nodiscard]] std::optional<Buffer>
  process_inputs(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::ml
