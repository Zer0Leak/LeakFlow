#include "leakflow/plugins/crypto_plot/poi_table_plot.hpp"

#include "leakflow/core/buffer.hpp"
#include "leakflow/plot/poi_table_view.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace {

namespace crypto = leakflow::plugins::crypto;

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

leakflow::Buffer poi_buffer(torch::Tensor result_unit0, const std::string& channels = "")
{
    std::vector<crypto::CorrelationPoiResult> results;
    results.push_back({0, std::move(result_unit0)});
    leakflow::Buffer buffer{leakflow::Caps(crypto::correlation_poi_caps_type)};
    if (!channels.empty()) {
        buffer.set_metadata("payload.leakage.channels", channels);
    }
    buffer.set_payload(std::make_shared<crypto::CorrelationPoiPayload>(std::move(results), std::string("correlation")));
    return buffer;
}

// [channel, k, 2] with the given per-channel (index, score) rows.
torch::Tensor index_score(std::vector<std::vector<std::pair<double, double>>> channels)
{
    std::vector<torch::Tensor> rows;
    for (const auto& channel : channels) {
        std::vector<torch::Tensor> pairs;
        for (const auto& [index, score] : channel) {
            pairs.push_back(torch::tensor({index, score}, torch::kFloat64));
        }
        rows.push_back(torch::stack(pairs, 0));
    }
    return torch::stack(rows, 0);
}

const std::vector<std::string> want_cols_ch0 = {"10", "11", "12"};

} // namespace

int main()
{
    const auto reference = index_score({{{10, 0.5}, {11, 0.4}, {12, 0.3}}, {{20, 0.2}, {21, 0.1}, {22, 0.05}}});
    const auto current = index_score({{{10, 0.9}, {11, 0.8}, {12, 0.7}}, {{20, 0.6}, {21, 0.5}, {22, 0.4}}});

    // --- Both inputs: sample-index columns, both rows filled, channel labels from metadata ---
    {
        auto view = std::make_shared<leakflow::plot::PoiTableView>();
        leakflow::plugins::crypto_plot::PoiTablePlot plot(view);
        leakflow::ElementInputs inputs;
        inputs.emplace("reference", poi_buffer(reference, "HW(m),HW(y)"));
        inputs.emplace("current", poi_buffer(current));
        const auto out = plot.process_inputs(std::move(inputs));
        if (!expect(!out.has_value(), "PoiTablePlot should be a sink")) {
            return 1;
        }
        const auto& snap = view->snapshots().front();
        if (!expect(snap.unit_ids == std::vector<std::int64_t>{0} && snap.groups.size() == 2,
                    "wrong units/groups")) {
            return 1;
        }
        if (!expect(snap.channel_labels == std::vector<std::string>{"HW(m)", "HW(y)"}, "channel labels wrong")) {
            return 1;
        }
        const auto& g0 = snap.groups[0]; // unit 0, channel 0
        if (!expect(g0.columns == want_cols_ch0, "channel 0 columns should be the sample indexes")) {
            return 1;
        }
        if (!expect(g0.reference == std::vector<std::string>{"0.500", "0.400", "0.300"}, "reference row wrong")) {
            return 1;
        }
        if (!expect(g0.current == std::vector<std::string>{"0.900", "0.800", "0.700"}, "current row wrong")) {
            return 1;
        }
        // Numeric fields the view sorts / highlights / plots from.
        if (!expect(g0.sample == std::vector<double>{10, 11, 12}, "sample keys should be the indexes")) {
            return 1;
        }
        if (!expect(g0.reference_values == std::vector<double>{0.5, 0.4, 0.3}, "reference values wrong")) {
            return 1;
        }
        if (!expect(g0.current_values == std::vector<double>{0.9, 0.8, 0.7}, "current values wrong")) {
            return 1;
        }
        if (!expect(g0.has_reference && g0.has_current, "both sides should be present")) {
            return 1;
        }
    }

    // --- Only reference present: current row shows "-" ---
    {
        auto view = std::make_shared<leakflow::plot::PoiTableView>();
        leakflow::plugins::crypto_plot::PoiTablePlot plot(view);
        leakflow::ElementInputs inputs;
        inputs.emplace("reference", poi_buffer(reference));
        const auto out = plot.process_inputs(std::move(inputs));
        (void)out;
        const auto& g0 = view->snapshots().front().groups[0];
        if (!expect(g0.reference == std::vector<std::string>{"0.500", "0.400", "0.300"}
                        && g0.current == std::vector<std::string>{"-", "-", "-"},
                    "absent current should be '-'")) {
            return 1;
        }
        if (!expect(g0.has_reference && !g0.has_current, "only reference should be marked present")) {
            return 1;
        }
        if (!expect(std::isnan(g0.current_values.at(0)), "absent current values should be NaN")) {
            return 1;
        }
    }

    // --- Score-only payload (last axis == 1): columns become 1..k ordinals ---
    {
        const auto scores = torch::tensor({{{0.5}, {0.4}, {0.3}}, {{0.2}, {0.1}, {0.05}}}, torch::kFloat64); // [2,3,1]
        auto view = std::make_shared<leakflow::plot::PoiTableView>();
        leakflow::plugins::crypto_plot::PoiTablePlot plot(view);
        leakflow::ElementInputs inputs;
        inputs.emplace("current", poi_buffer(scores));
        (void)plot.process_inputs(std::move(inputs));
        const auto& g0 = view->snapshots().front().groups[0];
        if (!expect(g0.columns == std::vector<std::string>{"1", "2", "3"}, "score-only columns should be 1..k")) {
            return 1;
        }
        if (!expect(g0.sample == std::vector<double>{1, 2, 3}, "score-only sample keys should be 1..k")) {
            return 1;
        }
        if (!expect(g0.current == std::vector<std::string>{"0.500", "0.400", "0.300"}, "score-only current wrong")) {
            return 1;
        }
    }

    std::cout << "poi_table_plot tests passed\n";
    return 0;
}
