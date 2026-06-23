#pragma once

#include "leakflow/core/payload.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace leakflow::plugins::crypto {

inline constexpr auto correlation_poi_caps_type = "leakflow/correlation-poi";

struct CorrelationPoiResult {
    std::uint16_t target_byte_index = 0;
    torch::Tensor result;
};

class CorrelationPoiPayload final : public Payload {
public:
    CorrelationPoiPayload(std::vector<CorrelationPoiResult> results, std::string score_name);

    [[nodiscard]] std::string type_name() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    [[nodiscard]] const std::vector<CorrelationPoiResult>& results() const;
    [[nodiscard]] const CorrelationPoiResult& result(std::size_t index) const;
    [[nodiscard]] std::size_t result_count() const;
    [[nodiscard]] const std::string& score_name() const;

private:
    std::vector<CorrelationPoiResult> results_;
    std::string score_name_;
};

} // namespace leakflow::plugins::crypto
