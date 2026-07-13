#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/summary_document.hpp"

#include <c10/core/ScalarType.h>

#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow::plugins::crypto {
namespace {

[[nodiscard]] std::string shape_to_string(c10::IntArrayRef values)
{
    std::vector<std::int64_t> vector_values(values.begin(), values.end());
    return summary_list_from_int_array(vector_values.data(), vector_values.size());
}

[[nodiscard]] std::string unit_result_summary(const CorrelationPoiResult& result)
{
    return "(unit: " + std::to_string(result.unit_index)
        + ", shape: " + shape_to_string(result.result.sizes()) + ")";
}

void validate_result(const CorrelationPoiResult& result)
{
    if (!result.result.defined()) {
        throw std::invalid_argument("CorrelationPoiPayload result tensor must be defined");
    }
    if (result.result.layout() != torch::kStrided) {
        throw std::invalid_argument("CorrelationPoiPayload result tensor must use strided layout");
    }
    if (result.result.dim() != 3 || (result.result.size(2) != 2 && result.result.size(2) != 1)) {
        // Last axis is (sample_index, score); [C,K,1] carries just the score (e.g. best-correlation
        // comparisons where there is no sample position).
        throw std::invalid_argument("CorrelationPoiPayload result tensor must have shape [C,K,2] or [C,K,1]");
    }
    if (!c10::isFloatingType(result.result.scalar_type())) {
        throw std::invalid_argument("CorrelationPoiPayload result tensor must be floating-point");
    }
}

} // namespace

CorrelationPoiPayload::CorrelationPoiPayload(std::vector<CorrelationPoiResult> results, std::string score_name)
    : results_(std::move(results))
    , score_name_(std::move(score_name))
{
    if (results_.empty()) {
        throw std::invalid_argument("CorrelationPoiPayload requires at least one result");
    }
    if (score_name_.empty()) {
        throw std::invalid_argument("CorrelationPoiPayload score name cannot be empty");
    }

    const auto field_count = results_.front().result.defined() && results_.front().result.dim() == 3
        ? results_.front().result.size(2)
        : std::int64_t{0};
    for (const auto& result : results_) {
        validate_result(result);
        if (result.unit_index < 0) {
            throw std::invalid_argument("CorrelationPoiPayload unit indexes must be non-negative");
        }
        if (result.result.size(2) != field_count) {
            throw std::invalid_argument("CorrelationPoiPayload units must use a consistent result field axis");
        }
    }
    std::set<std::int64_t> unit_indexes;
    for (const auto& result : results_) {
        if (!unit_indexes.insert(result.unit_index).second) {
            throw std::invalid_argument("CorrelationPoiPayload unit indexes must be unique");
        }
    }
}

std::string CorrelationPoiPayload::type_name() const
{
    return correlation_poi_caps_type;
}

std::string CorrelationPoiPayload::layout() const
{
    const auto field_axis = results_.front().result.size(2);
    if (field_axis == 1) {
        return "unit/channel/poi/[" + score_name_ + "]";
    }
    return "unit/channel/poi/[sample_index," + score_name_ + "]";
}

void CorrelationPoiPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field(
        "dtype", leakflow::base::torch_dtype_name(results_.front().result.scalar_type()), SummaryValueRole::TypeName);
    section.add_field("device", results_.front().result.device().str(), SummaryValueRole::Text);
    section.add_field("units", summary_integer(static_cast<std::int64_t>(results_.size())),
        SummaryValueRole::Number);
    section.add_field("score", score_name_, SummaryValueRole::Text);

    if (summary_level >= 1) {
        for (const auto& result : results_) {
            auto& field = section.add_field("result", unit_result_summary(result), SummaryValueRole::Size);
            if (summary_level >= 2) {
                field.add_child(
                    "unit",
                    summary_integer(static_cast<std::int64_t>(result.unit_index)),
                    SummaryValueRole::Number);
            }
        }
    }
}

const std::vector<CorrelationPoiResult>& CorrelationPoiPayload::results() const
{
    return results_;
}

const CorrelationPoiResult& CorrelationPoiPayload::result(std::size_t index) const
{
    return results_.at(index);
}

std::size_t CorrelationPoiPayload::unit_count() const
{
    return results_.size();
}

const std::string& CorrelationPoiPayload::score_name() const
{
    return score_name_;
}

} // namespace leakflow::plugins::crypto
