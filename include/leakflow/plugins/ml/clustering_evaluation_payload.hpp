#pragma once

#include "leakflow/core/payload.hpp"
#include "leakflow/ml/clustering_evaluation.hpp"

#include <map>
#include <string>

namespace leakflow::plugins::ml {

inline constexpr auto clustering_evaluation_caps_type =
    "leakflow/clustering-evaluation";

// Structured pipeline boundary for the generic numeric evaluator. The result
// remains authoritative; parameters are the bounded experiment/evaluator
// settings that a report or comparison table needs beside the metrics.
class ClusteringEvaluationPayload final : public Payload {
public:
  using Parameters = std::map<std::string, std::string>;

  explicit ClusteringEvaluationPayload(
      leakflow::ml::ClusteringEvaluationResult result,
      Parameters parameters = {});

  [[nodiscard]] std::string type_name() const override;
  [[nodiscard]] std::string layout() const override;
  void describe(SummarySection &section,
                std::int64_t summary_level) const override;

  [[nodiscard]] const leakflow::ml::ClusteringEvaluationResult &
  result() const noexcept;
  [[nodiscard]] const Parameters &parameters() const noexcept;

private:
  leakflow::ml::ClusteringEvaluationResult result_;
  Parameters parameters_;
};

} // namespace leakflow::plugins::ml
