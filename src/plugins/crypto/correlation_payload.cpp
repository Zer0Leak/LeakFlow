#include "leakflow/plugins/crypto/correlation_payload.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/summary_document.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace leakflow::plugins::crypto {

CorrelationPayload::CorrelationPayload(
    torch::Tensor grouped_correlation,
    std::vector<std::int64_t> unit_indexes,
    std::int64_t channel_count,
    std::int64_t feature_count,
    std::string score_name,
    std::int64_t observation_count)
    : grouped_correlation_(std::move(grouped_correlation))
    , unit_indexes_(std::move(unit_indexes))
    , channel_count_(channel_count)
    , feature_count_(feature_count)
    , score_name_(std::move(score_name))
    , observation_count_(observation_count)
{
    if (!grouped_correlation_.defined()) {
        throw std::invalid_argument("CorrelationPayload correlation tensor must be defined");
    }
    if (grouped_correlation_.dim() != 3) {
        throw std::invalid_argument("CorrelationPayload correlation tensor must have shape [unit,channel,feature]");
    }
    if (unit_indexes_.empty()) {
        throw std::invalid_argument("CorrelationPayload requires at least one unit");
    }
    if (std::ranges::any_of(unit_indexes_, [](const auto value) { return value < 0; })) {
        throw std::invalid_argument("CorrelationPayload unit indexes must be non-negative");
    }
    if (std::set<std::int64_t>(unit_indexes_.begin(), unit_indexes_.end()).size() != unit_indexes_.size()) {
        throw std::invalid_argument("CorrelationPayload unit indexes must be unique");
    }
    if (channel_count_ <= 0 || feature_count_ <= 0) {
        throw std::invalid_argument("CorrelationPayload channel and feature counts must be positive");
    }
    if (grouped_correlation_.size(0) != static_cast<std::int64_t>(unit_indexes_.size())
        || grouped_correlation_.size(1) != channel_count_
        || grouped_correlation_.size(2) != feature_count_) {
        throw std::invalid_argument(
            "CorrelationPayload correlation shape must match unit indexes, channel count, and feature count");
    }
    if (score_name_.empty()) {
        throw std::invalid_argument("CorrelationPayload score name cannot be empty");
    }
}

std::string CorrelationPayload::type_name() const
{
    return correlation_caps_type;
}

std::string CorrelationPayload::layout() const
{
    return "unit/channel/feature";
}

void CorrelationPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("units", summary_integer(static_cast<std::int64_t>(unit_indexes_.size())),
        SummaryValueRole::Number);
    section.add_field("channels", summary_integer(channel_count_), SummaryValueRole::Number);
    section.add_field("features", summary_integer(feature_count_), SummaryValueRole::Number);
    section.add_field("score", score_name_, SummaryValueRole::Text);
    section.add_field("observations", summary_integer(observation_count_), SummaryValueRole::Number);
    if (summary_level >= 2) {
        section.add_field(
            "dtype", leakflow::base::torch_dtype_name(grouped_correlation_.scalar_type()), SummaryValueRole::TypeName);
        section.add_field("device", grouped_correlation_.device().str(), SummaryValueRole::Text);
    }
}

const torch::Tensor& CorrelationPayload::grouped_correlation() const
{
    return grouped_correlation_;
}

const std::vector<std::int64_t>& CorrelationPayload::unit_indexes() const
{
    return unit_indexes_;
}

std::int64_t CorrelationPayload::unit_count() const
{
    return static_cast<std::int64_t>(unit_indexes_.size());
}

std::int64_t CorrelationPayload::channel_count() const
{
    return channel_count_;
}

std::int64_t CorrelationPayload::feature_count() const
{
    return feature_count_;
}

const std::string& CorrelationPayload::score_name() const
{
    return score_name_;
}

std::int64_t CorrelationPayload::observation_count() const
{
    return observation_count_;
}

} // namespace leakflow::plugins::crypto
