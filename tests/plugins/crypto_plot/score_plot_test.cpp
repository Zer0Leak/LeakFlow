#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plugins/crypto/attack_payload.hpp"
#include "leakflow/plugins/crypto_plot/score_plot.hpp"

#include <cmath>
#include <iostream>
#include <memory>
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

// Two units at observation count N: unit 0 succeeds, unit 1 fails.
leakflow::Buffer make_stats_buffer(std::int64_t observation_count)
{
    using namespace leakflow::plugins::crypto;
    const auto topk_score = torch::tensor(std::vector<double>{0.9, 0.4}).reshape({2, 1});
    const auto topk_margin = torch::tensor(std::vector<double>{0.5, 0.1}).reshape({2, 1});

    auto payload = std::make_shared<AttackStatsPayload>(
        torch::tensor(std::vector<std::int64_t>{0, 5}),   // true_rank
        torch::tensor(std::vector<std::int64_t>{42, 99}),  // true_guess
        torch::tensor(std::vector<double>{0.8, 0.2}),      // true_score
        torch::tensor(std::vector<std::int64_t>{42, 7}),   // top1_guess
        torch::tensor(std::vector<std::int64_t>{1, 2}),    // top2_guess
        torch::tensor(std::vector<double>{0.3, 0.1}),      // score_gap
        torch::tensor(std::vector<std::int64_t>{1, 0}).to(torch::kBool), // success
        torch::tensor(std::vector<std::int64_t>{0, 0}),    // best_channel
        torch::tensor(std::vector<std::int64_t>{100, 200}),// best_sample
        torch::tensor(std::vector<std::int64_t>{42, 7}).reshape({2, 1}), // topk_guess
        topk_score, torch::zeros({2, 1}, torch::kFloat64), topk_margin,  // score, margin(unused), relative_margin
        torch::zeros({2, 1}, torch::kFloat64), torch::zeros({2, 1}, torch::kFloat64),
        torch::zeros({2, 1}, torch::kFloat64),
        std::vector<std::int64_t>{0, 1},
        std::vector<std::string>{"HW(y)"},
        std::vector<std::string>{"relative_margin"});

    leakflow::Buffer buffer{leakflow::Caps(attack_stats_caps_type)};
    buffer.set_metadata("attack.observation_count", std::to_string(observation_count));
    buffer.set_payload(std::move(payload));
    return buffer;
}

} // namespace

int main()
{
    namespace crypto_plot = leakflow::plugins::crypto_plot;
    namespace plot = leakflow::plot;

    auto runtime = std::make_shared<plot::PlotRuntime>();
    crypto_plot::ScorePlot score_plot(runtime, "score");
    (void)score_plot.process(make_stats_buffer(10));
    (void)score_plot.process(make_stats_buffer(20));

    if (!expect(runtime->has_sessions(), "ScorePlot registered no plot session")) {
        return 1;
    }
    if (!expect(runtime->score_snapshots().size() == 1, "ScorePlot should own one score snapshot")) {
        return 1;
    }
    const auto& snapshot = runtime->score_snapshots().front();
    if (!expect(snapshot.element_name == "score" && snapshot.group == "cpa",
                "ScorePlot snapshot identity/group was wrong")) {
        return 1;
    }
    // Default metrics = [score, relative_margin] -> two stacked panels.
    if (!expect(snapshot.panels.size() == 2, "ScorePlot should build one panel per metric")) {
        return 1;
    }
    if (!expect(snapshot.panels[0].metric == "score" && snapshot.panels[1].metric == "relative_margin",
                "ScorePlot panels were wrong")) {
        return 1;
    }
    // Each panel: one series per unit, each with a point per buffer.
    const auto& score_panel = snapshot.panels[0];
    if (!expect(score_panel.series.size() == 2, "ScorePlot should build one series per unit")) {
        return 1;
    }
    if (!expect(score_panel.series[0].label == "byte 0" && score_panel.series[1].label == "byte 1",
                "ScorePlot series labels were wrong")) {
        return 1;
    }
    const auto& byte0 = score_panel.series[0];
    const auto& byte1 = score_panel.series[1];
    if (!expect(byte0.points.size() == 2 && byte1.points.size() == 2,
                "ScorePlot should append a point per streamed buffer")) {
        return 1;
    }
    // x = observation count from metadata.
    if (!expect(byte0.points[0].x == 10.0 && byte0.points[1].x == 20.0,
                "ScorePlot x should follow attack.observation_count")) {
        return 1;
    }
    // y = the metric value; score panel top-1 score.
    if (!expect(std::abs(byte0.points[0].y - 0.9) < 1.0e-5 && std::abs(byte1.points[0].y - 0.4) < 1.0e-5,
                "ScorePlot score values were wrong")) {
        return 1;
    }
    // Success encoded by marker: byte 0 succeeds (square), byte 1 fails (x/cross).
    if (!expect(byte0.points[0].marker == plot::TracePlotAnnotationMarker::Square,
                "ScorePlot success point should use the square marker")) {
        return 1;
    }
    if (!expect(byte1.points[0].marker == plot::TracePlotAnnotationMarker::Cross,
                "ScorePlot failure point should use the cross marker")) {
        return 1;
    }

    // Stop clears the runtime (Stop/Start recycle).
    runtime->clear();
    if (!expect(runtime->score_snapshots().empty() && !runtime->has_sessions(),
                "PlotRuntime clear should drop score snapshots")) {
        return 1;
    }

    return 0;
}
