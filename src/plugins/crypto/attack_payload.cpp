#include "leakflow/plugins/crypto/attack_payload.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/summary_document.hpp"

#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leakflow::plugins::crypto {
namespace {

void require_tensor(
    const torch::Tensor& tensor,
    std::string_view name,
    std::int64_t rank)
{
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " tensor must be defined");
    }
    if (tensor.layout() != torch::kStrided) {
        throw std::invalid_argument(std::string(name) + " tensor must use strided layout");
    }
    if (tensor.dim() != rank) {
        throw std::invalid_argument(std::string(name) + " tensor rank is wrong");
    }
}

void require_integral_tensor(const torch::Tensor& tensor, std::string_view name, std::int64_t rank)
{
    require_tensor(tensor, name, rank);
    if (!c10::isIntegralType(tensor.scalar_type(), true)) {
        throw std::invalid_argument(std::string(name) + " tensor must be integral");
    }
}

void require_floating_tensor(const torch::Tensor& tensor, std::string_view name, std::int64_t rank)
{
    require_tensor(tensor, name, rank);
    if (!c10::isFloatingType(tensor.scalar_type())) {
        throw std::invalid_argument(std::string(name) + " tensor must be floating-point");
    }
}

[[nodiscard]] std::string shape_to_string(c10::IntArrayRef values)
{
    std::vector<std::int64_t> vector_values(values.begin(), values.end());
    return summary_list_from_int_array(vector_values.data(), vector_values.size());
}

[[nodiscard]] std::string format_score(double value)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << value;
    return output.str();
}

[[nodiscard]] std::string unit_label(const std::vector<std::int64_t>& units, std::int64_t unit)
{
    if (unit >= 0 && static_cast<std::size_t>(unit) < units.size()) {
        return std::to_string(units[static_cast<std::size_t>(unit)]);
    }
    return std::to_string(unit);
}

[[nodiscard]] std::string channel_label(const std::vector<std::string>& channel_names, std::int64_t channel)
{
    if (channel >= 0 && static_cast<std::size_t>(channel) < channel_names.size()) {
        return channel_names[static_cast<std::size_t>(channel)];
    }
    return "channel_" + std::to_string(channel);
}

[[nodiscard]] const torch::Tensor& topk_metric_tensor(const AttackStatsPayload& payload, const std::string& metric)
{
    if (metric == "margin") {
        return payload.topk_margin();
    }
    if (metric == "relative_margin") {
        return payload.topk_relative_margin();
    }
    if (metric == "z_score") {
        return payload.topk_z_score();
    }
    if (metric == "robust_z_score") {
        return payload.topk_robust_z_score();
    }
    if (metric == "top_k_separation") {
        return payload.topk_separation();
    }
    throw std::invalid_argument("unknown attack confidence metric in payload summary");
}

void describe_attack_unit(
    SummarySection& section,
    std::int64_t unit,
    const AttackScoresPayload& payload)
{
    const auto best_guess = payload.best_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto best_score = payload.best_score().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto best_channel = payload.best_channel().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto best_sample = payload.best_sample().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto guess = best_guess[unit].item<std::int64_t>();
    const auto score = best_score[unit].item<double>();
    const auto channel = best_channel[unit].item<std::int64_t>();
    const auto sample = best_sample[unit].item<std::int64_t>();

    auto& field = section.add_field(
        "unit[" + std::to_string(unit) + "]",
        "best=" + std::to_string(guess) + " score=" + format_score(score),
        SummaryValueRole::Number);
    field.add_child("attack_unit", unit_label(payload.units(), unit), SummaryValueRole::Number);
    field.add_child("best_guess", summary_integer(guess), SummaryValueRole::Number);
    field.add_child("best_score", format_score(score), SummaryValueRole::Number);
    field.add_child("best_channel", channel_label(payload.channel_names(), channel), SummaryValueRole::Text);
    field.add_child("best_sample", summary_integer(sample), SummaryValueRole::Number);
}

void describe_stats_unit(
    SummarySection& section,
    std::int64_t unit,
    const AttackStatsPayload& payload,
    std::int64_t summary_level)
{
    const auto top1_tensor = payload.top1_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto gap_tensor = payload.score_gap().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto best_channel_tensor = payload.best_channel().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto best_sample_tensor = payload.best_sample().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto top1 = top1_tensor[unit].item<std::int64_t>();
    const auto gap = gap_tensor[unit].item<double>();
    const auto best_channel = best_channel_tensor[unit].item<std::int64_t>();
    const auto best_sample = best_sample_tensor[unit].item<std::int64_t>();

    auto header = "top1=" + std::to_string(top1) + " rel_margin=" + format_score(gap);
    if (payload.has_truth()) {
        const auto rank = payload.true_rank()->to(torch::kCPU).to(torch::kInt64).contiguous()[unit].item<std::int64_t>();
        header = "rank=" + std::to_string(rank) + " " + header;
    }
    auto& field = section.add_field(
        "unit[" + std::to_string(unit) + "]",
        header,
        SummaryValueRole::Number);
    field.add_child("attack_unit", unit_label(payload.units(), unit), SummaryValueRole::Number);
    if (payload.has_truth()) {
        const auto rank = payload.true_rank()->to(torch::kCPU).to(torch::kInt64).contiguous()[unit].item<std::int64_t>();
        const auto true_guess = payload.true_guess()->to(torch::kCPU).to(torch::kInt64).contiguous()[unit].item<std::int64_t>();
        const auto true_score = payload.true_score()->to(torch::kCPU).to(torch::kFloat64).contiguous()[unit].item<double>();
        field.add_child("true_rank", summary_integer(rank), SummaryValueRole::Number);
        field.add_child("true_guess", summary_integer(true_guess), SummaryValueRole::Number);
        field.add_child("true_score", format_score(true_score), SummaryValueRole::Number);
    }
    field.add_child("top1_guess", summary_integer(top1), SummaryValueRole::Number);
    field.add_child("best_channel", channel_label(payload.channel_names(), best_channel), SummaryValueRole::Text);
    field.add_child("best_sample", summary_integer(best_sample), SummaryValueRole::Number);
    field.add_child("relative_margin", format_score(gap), SummaryValueRole::Number);
    field.add_child("score_gap", format_score(gap), SummaryValueRole::Number);
    if (payload.has_truth()) {
        const auto success = payload.success()->to(torch::kCPU).to(torch::kBool).contiguous()[unit].item<bool>();
        field.add_child("success", summary_bool(success), SummaryValueRole::Boolean);
    }

    if (summary_level < 2) {
        return;
    }

    const auto topk_guess = payload.topk_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto topk_score = payload.topk_score().to(torch::kCPU).to(torch::kFloat64).contiguous();
    std::vector<torch::Tensor> selected_metric_tensors;
    selected_metric_tensors.reserve(payload.confidence_metrics().size());
    for (const auto& metric : payload.confidence_metrics()) {
        selected_metric_tensors.push_back(topk_metric_tensor(payload, metric).to(torch::kCPU).to(torch::kFloat64).contiguous());
    }

    for (std::int64_t rank_index = 0; rank_index < payload.top_k(); ++rank_index) {
        const auto guess = topk_guess[unit][rank_index].item<std::int64_t>();
        const auto score = topk_score[unit][rank_index].item<double>();
        auto& topk_field = field.add_child(
            "topk[" + std::to_string(rank_index) + "]",
            "guess=" + std::to_string(guess) + " score=" + format_score(score),
            SummaryValueRole::Number);
        topk_field.add_child("guess", summary_integer(guess), SummaryValueRole::Number);
        topk_field.add_child("score", format_score(score), SummaryValueRole::Number);
        for (std::size_t metric_index = 0; metric_index < payload.confidence_metrics().size(); ++metric_index) {
            topk_field.add_child(
                payload.confidence_metrics()[metric_index],
                format_score(selected_metric_tensors[metric_index][unit][rank_index].item<double>()),
                SummaryValueRole::Number);
        }
    }
}

} // namespace

AttackScoresPayload::AttackScoresPayload(
    torch::Tensor scores,
    torch::Tensor ranking,
    torch::Tensor best_guess,
    torch::Tensor best_guess_index,
    torch::Tensor best_score,
    torch::Tensor best_channel,
    torch::Tensor best_sample,
    torch::Tensor guess_values,
    std::optional<torch::Tensor> correlations,
    std::vector<std::int64_t> units,
    std::vector<std::string> channel_names,
    std::string score_method,
    std::string score_channels,
    std::int64_t observation_count,
    std::int64_t top_k)
    : scores_(std::move(scores))
    , ranking_(std::move(ranking))
    , best_guess_(std::move(best_guess))
    , best_guess_index_(std::move(best_guess_index))
    , best_score_(std::move(best_score))
    , best_channel_(std::move(best_channel))
    , best_sample_(std::move(best_sample))
    , guess_values_(std::move(guess_values))
    , correlations_(std::move(correlations))
    , units_(std::move(units))
    , channel_names_(std::move(channel_names))
    , score_method_(std::move(score_method))
    , score_channels_(std::move(score_channels))
    , observation_count_(observation_count)
    , top_k_(top_k)
{
    require_floating_tensor(scores_, "AttackScoresPayload scores", 2);
    require_integral_tensor(ranking_, "AttackScoresPayload ranking", 2);
    require_integral_tensor(best_guess_, "AttackScoresPayload best_guess", 1);
    require_integral_tensor(best_guess_index_, "AttackScoresPayload best_guess_index", 1);
    require_floating_tensor(best_score_, "AttackScoresPayload best_score", 1);
    require_integral_tensor(best_channel_, "AttackScoresPayload best_channel", 1);
    require_integral_tensor(best_sample_, "AttackScoresPayload best_sample", 1);
    require_integral_tensor(guess_values_, "AttackScoresPayload guess_values", 1);

    const auto unit_count = scores_.size(0);
    const auto guess_count = scores_.size(1);
    if (unit_count <= 0 || guess_count <= 0) {
        throw std::invalid_argument("AttackScoresPayload scores must have shape [U,G] with positive axes");
    }
    if (ranking_.size(0) != unit_count || ranking_.size(1) != guess_count) {
        throw std::invalid_argument("AttackScoresPayload ranking shape must match scores");
    }
    if (best_guess_.size(0) != unit_count || best_guess_index_.size(0) != unit_count
        || best_score_.size(0) != unit_count || best_channel_.size(0) != unit_count
        || best_sample_.size(0) != unit_count) {
        throw std::invalid_argument("AttackScoresPayload best tensors must have shape [U]");
    }
    if (guess_values_.size(0) != guess_count) {
        throw std::invalid_argument("AttackScoresPayload guess_values must have shape [G]");
    }
    if (correlations_) {
        require_floating_tensor(*correlations_, "AttackScoresPayload correlations", 4);
        if (correlations_->size(0) != unit_count || correlations_->size(1) != guess_count) {
            throw std::invalid_argument("AttackScoresPayload correlations must have shape [U,G,L,S]");
        }
    }
    if (score_method_.empty()) {
        throw std::invalid_argument("AttackScoresPayload score_method cannot be empty");
    }
    if (score_channels_.empty()) {
        throw std::invalid_argument("AttackScoresPayload score_channels cannot be empty");
    }
    if (observation_count_ < 0) {
        throw std::invalid_argument("AttackScoresPayload observation_count cannot be negative");
    }
    if (top_k_ <= 0) {
        throw std::invalid_argument("AttackScoresPayload top_k must be positive");
    }
}

std::string AttackScoresPayload::type_name() const
{
    return attack_scores_caps_type;
}

std::string AttackScoresPayload::layout() const
{
    auto value = std::string(
        "scores=unit/guess;ranking=unit/rank;best_guess=unit;best_guess_index=unit;"
        "best_score=unit;best_channel=unit;best_sample=unit;guess_values=guess");
    if (correlations_) {
        value += ";correlations=unit/guess/channel/sample";
    }
    return value;
}

void AttackScoresPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("dtype", leakflow::base::torch_dtype_name(scores_.scalar_type()), SummaryValueRole::TypeName);
    section.add_field("device", scores_.device().str(), SummaryValueRole::Text);
    section.add_field("scores", shape_to_string(scores_.sizes()), SummaryValueRole::Number);
    section.add_field("ranking", shape_to_string(ranking_.sizes()), SummaryValueRole::Number);
    section.add_field("score_method", score_method_, SummaryValueRole::Text);
    section.add_field("score_channels", score_channels_, SummaryValueRole::Text);
    section.add_field("observations", summary_integer(observation_count_), SummaryValueRole::Number);
    section.add_field("correlations", correlations_ ? shape_to_string(correlations_->sizes()) : std::string("not emitted"),
        correlations_ ? SummaryValueRole::Number : SummaryValueRole::Text);

    if (summary_level >= 1) {
        const auto shown = std::min<std::int64_t>(unit_count(), summary_level >= 3 ? unit_count() : top_k_);
        for (std::int64_t unit = 0; unit < shown; ++unit) {
            describe_attack_unit(section, unit, *this);
        }
    }
}

const torch::Tensor& AttackScoresPayload::scores() const { return scores_; }
const torch::Tensor& AttackScoresPayload::ranking() const { return ranking_; }
const torch::Tensor& AttackScoresPayload::best_guess() const { return best_guess_; }
const torch::Tensor& AttackScoresPayload::best_guess_index() const { return best_guess_index_; }
const torch::Tensor& AttackScoresPayload::best_score() const { return best_score_; }
const torch::Tensor& AttackScoresPayload::best_channel() const { return best_channel_; }
const torch::Tensor& AttackScoresPayload::best_sample() const { return best_sample_; }
const torch::Tensor& AttackScoresPayload::guess_values() const { return guess_values_; }
const std::optional<torch::Tensor>& AttackScoresPayload::correlations() const { return correlations_; }
const std::vector<std::int64_t>& AttackScoresPayload::units() const { return units_; }
const std::vector<std::string>& AttackScoresPayload::channel_names() const { return channel_names_; }
const std::string& AttackScoresPayload::score_method() const { return score_method_; }
const std::string& AttackScoresPayload::score_channels() const { return score_channels_; }
std::int64_t AttackScoresPayload::observation_count() const { return observation_count_; }
std::int64_t AttackScoresPayload::top_k() const { return top_k_; }
std::int64_t AttackScoresPayload::unit_count() const { return scores_.size(0); }
std::int64_t AttackScoresPayload::guess_count() const { return scores_.size(1); }

AttackStatsPayload::AttackStatsPayload(
    std::optional<torch::Tensor> true_rank,
    std::optional<torch::Tensor> true_guess,
    std::optional<torch::Tensor> true_score,
    torch::Tensor top1_guess,
    torch::Tensor top2_guess,
    torch::Tensor score_gap,
    std::optional<torch::Tensor> success,
    torch::Tensor best_channel,
    torch::Tensor best_sample,
    torch::Tensor topk_guess,
    torch::Tensor topk_score,
    torch::Tensor topk_margin,
    torch::Tensor topk_relative_margin,
    torch::Tensor topk_z_score,
    torch::Tensor topk_robust_z_score,
    torch::Tensor topk_separation,
    std::vector<std::int64_t> units,
    std::vector<std::string> channel_names,
    std::vector<std::string> confidence_metrics)
    : true_rank_(std::move(true_rank))
    , true_guess_(std::move(true_guess))
    , true_score_(std::move(true_score))
    , top1_guess_(std::move(top1_guess))
    , top2_guess_(std::move(top2_guess))
    , score_gap_(std::move(score_gap))
    , success_(std::move(success))
    , best_channel_(std::move(best_channel))
    , best_sample_(std::move(best_sample))
    , topk_guess_(std::move(topk_guess))
    , topk_score_(std::move(topk_score))
    , topk_margin_(std::move(topk_margin))
    , topk_relative_margin_(std::move(topk_relative_margin))
    , topk_z_score_(std::move(topk_z_score))
    , topk_robust_z_score_(std::move(topk_robust_z_score))
    , topk_separation_(std::move(topk_separation))
    , units_(std::move(units))
    , channel_names_(std::move(channel_names))
    , confidence_metrics_(std::move(confidence_metrics))
{
    // Truth-derived tensors are all-or-nothing: present only when AttackStats had
    // a connected truth input.
    const bool has_truth = true_rank_.has_value() || true_guess_.has_value()
        || true_score_.has_value() || success_.has_value();
    if (has_truth && (!true_rank_ || !true_guess_ || !true_score_ || !success_)) {
        throw std::invalid_argument(
            "AttackStatsPayload truth tensors (true_rank, true_guess, true_score, success) "
            "must all be present or all be absent");
    }

    require_integral_tensor(top1_guess_, "AttackStatsPayload top1_guess", 1);
    require_integral_tensor(top2_guess_, "AttackStatsPayload top2_guess", 1);
    require_floating_tensor(score_gap_, "AttackStatsPayload score_gap", 1);
    if (has_truth) {
        require_integral_tensor(*true_rank_, "AttackStatsPayload true_rank", 1);
        require_integral_tensor(*true_guess_, "AttackStatsPayload true_guess", 1);
        require_floating_tensor(*true_score_, "AttackStatsPayload true_score", 1);
        require_tensor(*success_, "AttackStatsPayload success", 1);
        if (success_->scalar_type() != torch::kBool) {
            throw std::invalid_argument("AttackStatsPayload success tensor must have dtype bool");
        }
    }
    require_integral_tensor(best_channel_, "AttackStatsPayload best_channel", 1);
    require_integral_tensor(best_sample_, "AttackStatsPayload best_sample", 1);
    require_integral_tensor(topk_guess_, "AttackStatsPayload topk_guess", 2);
    require_floating_tensor(topk_score_, "AttackStatsPayload topk_score", 2);
    require_floating_tensor(topk_margin_, "AttackStatsPayload topk_margin", 2);
    require_floating_tensor(topk_relative_margin_, "AttackStatsPayload topk_relative_margin", 2);
    require_floating_tensor(topk_z_score_, "AttackStatsPayload topk_z_score", 2);
    require_floating_tensor(topk_robust_z_score_, "AttackStatsPayload topk_robust_z_score", 2);
    require_floating_tensor(topk_separation_, "AttackStatsPayload topk_separation", 2);

    const auto unit_count = top1_guess_.size(0);
    if (unit_count <= 0) {
        throw std::invalid_argument("AttackStatsPayload tensors must have positive unit count");
    }
    if (top2_guess_.size(0) != unit_count || score_gap_.size(0) != unit_count
        || best_channel_.size(0) != unit_count || best_sample_.size(0) != unit_count) {
        throw std::invalid_argument("AttackStatsPayload tensors must all have shape [U]");
    }
    if (has_truth
        && (true_rank_->size(0) != unit_count || true_guess_->size(0) != unit_count
            || true_score_->size(0) != unit_count || success_->size(0) != unit_count)) {
        throw std::invalid_argument("AttackStatsPayload truth tensors must all have shape [U]");
    }
    const auto top_k = topk_guess_.size(1);
    if (top_k <= 0) {
        throw std::invalid_argument("AttackStatsPayload top-K tensors must have positive K");
    }
    if (topk_guess_.size(0) != unit_count || topk_score_.size(0) != unit_count
        || topk_margin_.size(0) != unit_count || topk_relative_margin_.size(0) != unit_count
        || topk_z_score_.size(0) != unit_count || topk_robust_z_score_.size(0) != unit_count
        || topk_separation_.size(0) != unit_count || topk_score_.size(1) != top_k
        || topk_margin_.size(1) != top_k || topk_relative_margin_.size(1) != top_k
        || topk_z_score_.size(1) != top_k || topk_robust_z_score_.size(1) != top_k
        || topk_separation_.size(1) != top_k) {
        throw std::invalid_argument("AttackStatsPayload top-K tensors must all have shape [U,K]");
    }
}

std::string AttackStatsPayload::type_name() const
{
    return attack_stats_caps_type;
}

std::string AttackStatsPayload::layout() const
{
    auto value = std::string();
    if (has_truth()) {
        value = "true_rank=unit;true_guess=unit;true_score=unit;success=unit;";
    }
    value +=
        "top1_guess=unit;top2_guess=unit;score_gap=unit;best_channel=unit;best_sample=unit;"
        "topk_guess=unit/rank;topk_score=unit/rank;topk_margin=unit/rank;"
        "topk_relative_margin=unit/rank;topk_z_score=unit/rank;"
        "topk_robust_z_score=unit/rank;topk_separation=unit/rank";
    return value;
}

void AttackStatsPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("device", top1_guess_.device().str(), SummaryValueRole::Text);
    section.add_field("units", summary_integer(unit_count()), SummaryValueRole::Number);
    section.add_field("has_truth", summary_bool(has_truth()), SummaryValueRole::Boolean);
    section.add_field("top_k", summary_integer(top_k()), SummaryValueRole::Number);
    section.add_field("score_gap", "relative_margin", SummaryValueRole::Text);
    section.add_field("confidence_metrics", summary_list(confidence_metrics_), SummaryValueRole::Text);

    if (has_truth()) {
        section.add_field("rank_base", "1", SummaryValueRole::Number);
        const auto success_cpu = success_->to(torch::kCPU).to(torch::kInt64);
        section.add_field("successes", summary_integer(success_cpu.sum().item<std::int64_t>()), SummaryValueRole::Number);
    }

    if (summary_level >= 1) {
        const auto shown = summary_level >= 3 ? unit_count() : std::min<std::int64_t>(unit_count(), 8);
        for (std::int64_t unit = 0; unit < shown; ++unit) {
            describe_stats_unit(section, unit, *this, summary_level);
        }
    }
}

bool AttackStatsPayload::has_truth() const { return true_rank_.has_value(); }
const std::optional<torch::Tensor>& AttackStatsPayload::true_rank() const { return true_rank_; }
const std::optional<torch::Tensor>& AttackStatsPayload::true_guess() const { return true_guess_; }
const std::optional<torch::Tensor>& AttackStatsPayload::true_score() const { return true_score_; }
const torch::Tensor& AttackStatsPayload::top1_guess() const { return top1_guess_; }
const torch::Tensor& AttackStatsPayload::top2_guess() const { return top2_guess_; }
const torch::Tensor& AttackStatsPayload::score_gap() const { return score_gap_; }
const std::optional<torch::Tensor>& AttackStatsPayload::success() const { return success_; }
const torch::Tensor& AttackStatsPayload::best_channel() const { return best_channel_; }
const torch::Tensor& AttackStatsPayload::best_sample() const { return best_sample_; }
const torch::Tensor& AttackStatsPayload::topk_guess() const { return topk_guess_; }
const torch::Tensor& AttackStatsPayload::topk_score() const { return topk_score_; }
const torch::Tensor& AttackStatsPayload::topk_margin() const { return topk_margin_; }
const torch::Tensor& AttackStatsPayload::topk_relative_margin() const { return topk_relative_margin_; }
const torch::Tensor& AttackStatsPayload::topk_z_score() const { return topk_z_score_; }
const torch::Tensor& AttackStatsPayload::topk_robust_z_score() const { return topk_robust_z_score_; }
const torch::Tensor& AttackStatsPayload::topk_separation() const { return topk_separation_; }
const std::vector<std::int64_t>& AttackStatsPayload::units() const { return units_; }
const std::vector<std::string>& AttackStatsPayload::channel_names() const { return channel_names_; }
const std::vector<std::string>& AttackStatsPayload::confidence_metrics() const { return confidence_metrics_; }
std::int64_t AttackStatsPayload::unit_count() const { return top1_guess_.size(0); }
std::int64_t AttackStatsPayload::top_k() const { return topk_guess_.size(1); }

} // namespace leakflow::plugins::crypto
