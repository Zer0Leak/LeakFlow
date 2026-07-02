#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/plugins/crypto/attack_payload.hpp"
#include "leakflow/plugins/crypto/attack_stats_to_plot_annotations.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

// Two units, top_k=1. With truth: unit 0 succeeds, unit 1 fails. Without truth all
// truth tensors are absent. The converter encodes success in the marker shape and
// keeps the annotation height (norm_value) positive in every case.
leakflow::Buffer make_stats_buffer(bool with_truth)
{
    using namespace leakflow::plugins::crypto;
    const auto topk_score = torch::tensor(std::vector<double>{0.9, 0.4}).reshape({2, 1});
    const auto topk_zeros = torch::zeros({2, 1}, torch::TensorOptions().dtype(torch::kFloat64));

    std::optional<torch::Tensor> true_rank;
    std::optional<torch::Tensor> true_guess;
    std::optional<torch::Tensor> true_score;
    std::optional<torch::Tensor> success;
    if (with_truth) {
        true_rank = torch::tensor(std::vector<std::int64_t>{0, 5});
        true_guess = torch::tensor(std::vector<std::int64_t>{42, 99});
        true_score = torch::tensor(std::vector<double>{0.8, 0.2});
        success = torch::tensor(std::vector<std::int64_t>{1, 0}).to(torch::kBool);
    }

    auto payload = std::make_shared<AttackStatsPayload>(
        true_rank, true_guess, true_score,
        torch::tensor(std::vector<std::int64_t>{42, 7}),    // top1_guess
        torch::tensor(std::vector<std::int64_t>{1, 2}),     // top2_guess
        torch::tensor(std::vector<double>{0.3, 0.1}),       // score_gap (relative margin)
        success,
        torch::tensor(std::vector<std::int64_t>{0, 0}),     // best_channel
        torch::tensor(std::vector<std::int64_t>{100, 200}), // best_sample
        torch::tensor(std::vector<std::int64_t>{42, 7}).reshape({2, 1}), // topk_guess
        topk_score, topk_zeros, topk_zeros, topk_zeros, topk_zeros, topk_zeros,
        std::vector<std::int64_t>{0, 1},
        std::vector<std::string>{"HW(y)"},
        std::vector<std::string>{"relative_margin"});

    leakflow::Buffer buffer{leakflow::Caps(attack_stats_caps_type)};
    buffer.set_payload(std::move(payload));
    return buffer;
}

} // namespace

int main()
{
    using namespace leakflow::plugins::crypto;
    namespace base = leakflow::base;

    {
        AttackStatsToPlotAnnotations converter("ann");
        const auto output = converter.process(make_stats_buffer(/*with_truth=*/true));
        if (!expect(output.has_value(), "converter produced no output")) {
            return 1;
        }
        const auto annotations = output->payload_as<base::PlotAnnotationPayload>();
        if (!expect(annotations != nullptr, "output was not a PlotAnnotationPayload")) {
            return 1;
        }
        if (!expect(annotations->annotation_count() == 2, "expected one annotation per unit")) {
            return 1;
        }
        const auto& success_annotation = annotations->annotation(0);
        const auto& failure_annotation = annotations->annotation(1);
        if (!expect(success_annotation.marker == "square", "success annotation marker should be square")) {
            return 1;
        }
        if (!expect(failure_annotation.marker == "x", "failure annotation marker should be x")) {
            return 1;
        }
        // The value is no longer negated on failure: both stay positive (height = score).
        if (!expect(success_annotation.norm_value && *success_annotation.norm_value > 0.0,
                    "success annotation norm_value should be positive")) {
            return 1;
        }
        if (!expect(failure_annotation.norm_value && *failure_annotation.norm_value > 0.0,
                    "failure annotation norm_value should stay positive (not negated)")) {
            return 1;
        }
    }

    {
        AttackStatsToPlotAnnotations converter("ann");
        const auto output = converter.process(make_stats_buffer(/*with_truth=*/false));
        if (!expect(output.has_value(), "no-truth converter produced no output")) {
            return 1;
        }
        const auto annotations = output->payload_as<base::PlotAnnotationPayload>();
        if (!expect(annotations != nullptr, "no-truth output was not a PlotAnnotationPayload")) {
            return 1;
        }
        const auto& annotation = annotations->annotation(0);
        if (!expect(annotation.marker == "circle", "no-truth annotation marker should be circle")) {
            return 1;
        }
        if (!expect(annotation.norm_value && *annotation.norm_value > 0.0,
                    "no-truth annotation norm_value should be positive")) {
            return 1;
        }
    }

    return 0;
}
