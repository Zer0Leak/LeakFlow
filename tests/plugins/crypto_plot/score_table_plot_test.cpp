#include "leakflow/plot/plot_runtime.hpp"
#include "leakflow/plot/table_view.hpp"
#include "leakflow/plugins/crypto/attack_payload.hpp"
#include "leakflow/plugins/crypto_plot/score_table_plot.hpp"

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

// Two units, top_k=2. Unit 0: guesses [42,1] scores [0.9,0.5], true key 42 (rank 0 ->
// success). Unit 1: guesses [7,2] scores [0.4,0.35], true key 99 (rank 5 -> failure,
// and not in the visible top_k).
leakflow::Buffer make_stats_buffer(std::int64_t observation_count)
{
    using namespace leakflow::plugins::crypto;
    const auto topk_score = torch::tensor(std::vector<double>{0.9, 0.5, 0.4, 0.35}).reshape({2, 2});
    const auto topk_zeros = torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kFloat64));
    auto payload = std::make_shared<AttackStatsPayload>(
        torch::tensor(std::vector<std::int64_t>{0, 5}),    // true_rank
        torch::tensor(std::vector<std::int64_t>{42, 99}),  // true_guess
        torch::tensor(std::vector<double>{0.9, 0.2}),      // true_score
        torch::tensor(std::vector<std::int64_t>{42, 7}),   // top1_guess
        torch::tensor(std::vector<std::int64_t>{1, 2}),    // top2_guess
        torch::tensor(std::vector<double>{0.4, 0.05}),     // score_gap
        torch::tensor(std::vector<std::int64_t>{1, 0}).to(torch::kBool),          // success
        torch::tensor(std::vector<std::int64_t>{0, 0}),    // best_channel
        torch::tensor(std::vector<std::int64_t>{100, 200}),// best_sample
        torch::tensor(std::vector<std::int64_t>{42, 1, 7, 2}).reshape({2, 2}),    // topk_guess
        topk_score, topk_zeros, topk_zeros, topk_zeros, topk_zeros, topk_zeros,
        std::vector<std::int64_t>{0, 1}, std::vector<std::string>{"HW(y)"},
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

    // ScoreTablePlot fills a TableView registered with the PlotRuntime (registry).
    auto runtime = std::make_shared<plot::PlotRuntime>();
    auto view = std::make_shared<plot::TableView>();
    runtime->add_view(view);
    crypto_plot::ScoreTablePlot table_plot(view, "table");
    (void)table_plot.process(make_stats_buffer(10));

    if (!expect(runtime->has_sessions(), "TableView registered with the runtime should count as a session")) {
        return 1;
    }
    if (!expect(view->snapshots().size() == 1, "ScoreTablePlot should own one table snapshot")) {
        return 1;
    }
    const auto& snap = view->snapshots().front();
    if (!expect(snap.element_name == "table" && snap.group == "cpa", "table snapshot identity/group was wrong")) {
        return 1;
    }
    // Columns = rank + one per unit; rows = top_k ranked candidates; one history frame.
    if (!expect(snap.columns.size() == 3 && snap.columns[0] == "rank/unit" && snap.columns[1] == "0" &&
                    snap.columns[2] == "1",
                "table columns were wrong")) {
        return 1;
    }
    if (!expect(snap.frames.size() == 1, "default max_history should keep a single frame")) {
        return 1;
    }
    const auto& frame = snap.frames.back();
    if (!expect(frame.caption == "N = 10", "frame caption should carry the observation count")) {
        return 1;
    }
    if (!expect(frame.rows.size() == 2, "one row per ranked candidate (top_k=2)")) {
        return 1;
    }
    if (!expect(frame.rows[0][0].text == "#1" && frame.rows[1][0].text == "#2", "rank column was wrong")) {
        return 1;
    }
    // Sort by score (default): row 0 is each unit's best. Unit 0's best (0x2a) is the
    // correct key -> tinted + emphasized; unit 1's best (0x07) is a wrong guess ->
    // emphasized but not tinted (true key 0x63 is outside the visible top_k).
    const auto& u0_best = frame.rows[0][1];
    const auto& u1_best = frame.rows[0][2];
    if (!expect(u0_best.text.rfind("2a", 0) == 0 && u0_best.emphasize && u0_best.tint.has_value(),
                "unit 0 rank-1 cell should be the correct key (tinted + emphasized)")) {
        return 1;
    }
    if (!expect(u1_best.text.rfind("07", 0) == 0 && u1_best.emphasize && !u1_best.tint.has_value(),
                "unit 1 rank-1 cell should be emphasized but not tinted (wrong best guess)")) {
        return 1;
    }
    if (!expect(!frame.rows[1][1].tint.has_value() && !frame.rows[1][1].emphasize,
                "unit 0 rank-2 cell (0x01) is neither correct nor the leader")) {
        return 1;
    }

    // Sort by guess: unit 0's candidates reorder to [0x01, 0x2a], so the correct/leader
    // cell moves to row 1 (position 0 stays emphasized; the true key stays tinted).
    {
        auto view2 = std::make_shared<plot::TableView>();
        crypto_plot::ScoreTablePlot table_plot2(view2, "table2");
        table_plot2.set_property("sort", std::string("guess"));
        (void)table_plot2.process(make_stats_buffer(10));
        const auto& f = view2->snapshots().front().frames.back();
        if (!expect(f.rows[0][1].text.rfind("01", 0) == 0, "guess sort should put 0x01 first for unit 0")) {
            return 1;
        }
        if (!expect(f.rows[1][1].text.rfind("2a", 0) == 0 && f.rows[1][1].emphasize && f.rows[1][1].tint.has_value(),
                    "guess sort should move the correct/leader cell to row 2")) {
            return 1;
        }
    }

    // max_history: keep the last N frames; a further push trims the oldest.
    {
        auto view3 = std::make_shared<plot::TableView>();
        crypto_plot::ScoreTablePlot table_plot3(view3, "table3");
        table_plot3.set_property("max_history", std::int64_t{3});
        for (std::int64_t n : {10, 20, 30, 40}) {
            (void)table_plot3.process(make_stats_buffer(n));
        }
        const auto& s = view3->snapshots().front();
        if (!expect(s.frames.size() == 3, "max_history=3 should keep the last three frames")) {
            return 1;
        }
        if (!expect(s.frames.front().caption == "N = 20" && s.frames.back().caption == "N = 40",
                    "history should drop the oldest frame")) {
            return 1;
        }
    }

    // Stop clears the runtime; the clear cascades to the TableView.
    runtime->clear();
    if (!expect(view->snapshots().empty() && !runtime->has_sessions(),
                "PlotRuntime clear should cascade to the TableView")) {
        return 1;
    }

    return 0;
}
