#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/summary_document.hpp"

#include <c10/core/ScalarType.h>

#include <sstream>
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

[[nodiscard]] std::string byte_result_summary(const CorrelationPoiResult& result)
{
    return "(byte_index: " + std::to_string(result.target_byte_index)
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
    if (result.result.dim() != 3 || result.result.size(2) != 2) {
        throw std::invalid_argument("CorrelationPoiPayload result tensor must have shape [C,K,2]");
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

    for (const auto& result : results_) {
        validate_result(result);
    }
}

std::string CorrelationPoiPayload::type_name() const
{
    return correlation_poi_caps_type;
}

void CorrelationPoiPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("result groups", summary_integer(static_cast<std::int64_t>(results_.size())),
        SummaryValueRole::Number);
    section.add_field("score", score_name_, SummaryValueRole::Text);

    if (summary_level >= 1) {
        for (const auto& result : results_) {
            auto& field = section.add_field("result", byte_result_summary(result), SummaryValueRole::Size);
            if (summary_level >= 2) {
                field.add_child(
                    "byte_index",
                    summary_integer(static_cast<std::int64_t>(result.target_byte_index)),
                    SummaryValueRole::Number);
                field.add_child(
                    "dtype", leakflow::base::torch_dtype_name(result.result.scalar_type()), SummaryValueRole::TypeName);
                field.add_child("device", result.result.device().str(), SummaryValueRole::Text);
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

std::size_t CorrelationPoiPayload::result_count() const
{
    return results_.size();
}

const std::string& CorrelationPoiPayload::score_name() const
{
    return score_name_;
}

} // namespace leakflow::plugins::crypto
