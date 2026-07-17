#pragma once

#include "leakflow/core/payload.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace leakflow::plugins::crypto {

inline constexpr auto attack_scores_caps_type = "leakflow/attack-scores";
inline constexpr auto attack_stats_caps_type = "leakflow/attack-stats";

class AttackScoresPayload final : public Payload {
public:
    AttackScoresPayload(
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
        std::int64_t top_k);

    [[nodiscard]] std::string type_name() const override;
    [[nodiscard]] std::string layout() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    [[nodiscard]] const torch::Tensor& scores() const;
    [[nodiscard]] const torch::Tensor& ranking() const;
    [[nodiscard]] const torch::Tensor& best_guess() const;
    [[nodiscard]] const torch::Tensor& best_guess_index() const;
    [[nodiscard]] const torch::Tensor& best_score() const;
    [[nodiscard]] const torch::Tensor& best_channel() const;
    [[nodiscard]] const torch::Tensor& best_sample() const;
    [[nodiscard]] const torch::Tensor& guess_values() const;
    [[nodiscard]] const std::optional<torch::Tensor>& correlations() const;
    [[nodiscard]] const std::vector<std::int64_t>& units() const;
    [[nodiscard]] const std::vector<std::string>& channel_names() const;
    [[nodiscard]] const std::string& score_method() const;
    [[nodiscard]] const std::string& score_channels() const;
    [[nodiscard]] std::int64_t observation_count() const;
    [[nodiscard]] std::int64_t top_k() const;
    [[nodiscard]] std::int64_t unit_count() const;
    [[nodiscard]] std::int64_t guess_count() const;

private:
    torch::Tensor scores_;
    torch::Tensor ranking_;
    torch::Tensor best_guess_;
    torch::Tensor best_guess_index_;
    torch::Tensor best_score_;
    torch::Tensor best_channel_;
    torch::Tensor best_sample_;
    torch::Tensor guess_values_;
    std::optional<torch::Tensor> correlations_;
    std::vector<std::int64_t> units_;
    std::vector<std::string> channel_names_;
    std::string score_method_;
    std::string score_channels_;
    std::int64_t observation_count_ = 0;
    std::int64_t top_k_ = 5;
};

class AttackStatsPayload final : public Payload {
public:
    AttackStatsPayload(
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
        std::vector<std::string> confidence_metrics);

    [[nodiscard]] std::string type_name() const override;
    [[nodiscard]] std::string layout() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    // Known-key (truth) diagnostics are only available when AttackStats was given
    // a connected truth input. Without it, GE/PGE-style fields are absent.
    [[nodiscard]] bool has_truth() const;
    [[nodiscard]] const std::optional<torch::Tensor>& true_rank() const;
    [[nodiscard]] const std::optional<torch::Tensor>& true_guess() const;
    [[nodiscard]] const std::optional<torch::Tensor>& true_score() const;
    [[nodiscard]] const torch::Tensor& top1_guess() const;
    [[nodiscard]] const torch::Tensor& top2_guess() const;
    [[nodiscard]] const torch::Tensor& score_gap() const;
    [[nodiscard]] const std::optional<torch::Tensor>& success() const;
    [[nodiscard]] const torch::Tensor& best_channel() const;
    [[nodiscard]] const torch::Tensor& best_sample() const;
    [[nodiscard]] const torch::Tensor& topk_guess() const;
    [[nodiscard]] const torch::Tensor& topk_score() const;
    [[nodiscard]] const torch::Tensor& topk_margin() const;
    [[nodiscard]] const torch::Tensor& topk_relative_margin() const;
    [[nodiscard]] const torch::Tensor& topk_z_score() const;
    [[nodiscard]] const torch::Tensor& topk_robust_z_score() const;
    [[nodiscard]] const torch::Tensor& topk_separation() const;
    [[nodiscard]] const std::vector<std::int64_t>& units() const;
    [[nodiscard]] const std::vector<std::string>& channel_names() const;
    [[nodiscard]] const std::vector<std::string>& confidence_metrics() const;
    [[nodiscard]] std::int64_t unit_count() const;
    [[nodiscard]] std::int64_t top_k() const;

private:
    std::optional<torch::Tensor> true_rank_;
    std::optional<torch::Tensor> true_guess_;
    std::optional<torch::Tensor> true_score_;
    torch::Tensor top1_guess_;
    torch::Tensor top2_guess_;
    torch::Tensor score_gap_;
    std::optional<torch::Tensor> success_;
    torch::Tensor best_channel_;
    torch::Tensor best_sample_;
    torch::Tensor topk_guess_;
    torch::Tensor topk_score_;
    torch::Tensor topk_margin_;
    torch::Tensor topk_relative_margin_;
    torch::Tensor topk_z_score_;
    torch::Tensor topk_robust_z_score_;
    torch::Tensor topk_separation_;
    std::vector<std::int64_t> units_;
    std::vector<std::string> channel_names_;
    std::vector<std::string> confidence_metrics_;
};

} // namespace leakflow::plugins::crypto
