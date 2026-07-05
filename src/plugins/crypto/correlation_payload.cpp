#include "leakflow/plugins/crypto/correlation_payload.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/summary_document.hpp"

#include <stdexcept>
#include <utility>

namespace leakflow::plugins::crypto {

CorrelationPayload::CorrelationPayload(
    torch::Tensor grouped_correlation,
    std::vector<std::uint16_t> byte_indexes,
    std::int64_t channel_count,
    std::int64_t feature_count,
    std::string score_name,
    std::int64_t observation_count)
    : grouped_correlation_(std::move(grouped_correlation))
    , byte_indexes_(std::move(byte_indexes))
    , channel_count_(channel_count)
    , feature_count_(feature_count)
    , score_name_(std::move(score_name))
    , observation_count_(observation_count)
{
    if (!grouped_correlation_.defined()) {
        throw std::invalid_argument("CorrelationPayload correlation tensor must be defined");
    }
    if (grouped_correlation_.dim() != 3) {
        throw std::invalid_argument("CorrelationPayload correlation tensor must have shape [byte,channel,feature]");
    }
    if (byte_indexes_.empty()) {
        throw std::invalid_argument("CorrelationPayload requires at least one byte group");
    }
    if (score_name_.empty()) {
        throw std::invalid_argument("CorrelationPayload score name cannot be empty");
    }
}

std::string CorrelationPayload::type_name() const
{
    return correlation_caps_type;
}

void CorrelationPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("byte groups", summary_integer(static_cast<std::int64_t>(byte_indexes_.size())),
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

const std::vector<std::uint16_t>& CorrelationPayload::byte_indexes() const
{
    return byte_indexes_;
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
