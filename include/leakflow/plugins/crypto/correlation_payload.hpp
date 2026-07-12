#pragma once

#include "leakflow/core/payload.hpp"

#include <cstdint>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace leakflow::plugins::crypto {

inline constexpr auto correlation_caps_type = "leakflow/correlation";

// The correlation grid produced by PearsonCorrelator, before PoI selection. It holds
// the correlation of every feature against every target, grouped as
// [unit, channel, feature], plus the unit indexes and score name the target
// tensor represented. PoiSelect turns this into a CorrelationPoiPayload by picking
// top-k features per (unit, channel) -- a pure, stateless re-selection, which is why
// the accumulation (PearsonCorrelator, stateful/non-replayable) and the selection
// (PoiSelect, stateless/replayable) live in separate elements.
class CorrelationPayload final : public Payload {
public:
    CorrelationPayload(
        torch::Tensor grouped_correlation,
        std::vector<std::int64_t> unit_indexes,
        std::int64_t channel_count,
        std::int64_t feature_count,
        std::string score_name,
        std::int64_t observation_count);

    [[nodiscard]] std::string type_name() const override;
    [[nodiscard]] std::string layout() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    // [unit, channel, feature] correlation.
    [[nodiscard]] const torch::Tensor& grouped_correlation() const;
    [[nodiscard]] const std::vector<std::int64_t>& unit_indexes() const;
    [[nodiscard]] std::int64_t unit_count() const;
    [[nodiscard]] std::int64_t channel_count() const;
    [[nodiscard]] std::int64_t feature_count() const;
    [[nodiscard]] const std::string& score_name() const;
    [[nodiscard]] std::int64_t observation_count() const;

private:
    torch::Tensor grouped_correlation_;
    std::vector<std::int64_t> unit_indexes_;
    std::int64_t channel_count_ = 1;
    std::int64_t feature_count_ = 0;
    std::string score_name_;
    std::int64_t observation_count_ = 0;
};

} // namespace leakflow::plugins::crypto
