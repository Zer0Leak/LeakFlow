#include "leakflow/plugins/plot/heatmap_plot.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/plot/heatmap_view.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <torch/torch.h>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

leakflow::Buffer torch_buffer(torch::Tensor tensor)
{
    auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    leakflow::Buffer buffer{payload.caps()};
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    return buffer;
}

torch::Tensor snapshot_matrix(const leakflow::plot::HeatmapSnapshot& s, std::int64_t unit = 0)
{
    return torch::from_blob(
        const_cast<double*>(s.data.data()) + unit * s.rows * s.cols, {s.rows, s.cols}, torch::kFloat64)
        .clone();
}

} // namespace

int main()
{
    const auto confusion = 2.0 * torch::eye(3, torch::kFloat64); // diagonal "confusion"

    // --- Row normalization: diagonal counts -> identity, scale [0,1] ---
    {
        auto view = std::make_shared<leakflow::plot::HeatmapView>();
        leakflow::plugins::plot::HeatmapPlot plot(view);
        plot.set_property("normalize", std::string("row"));
        const auto out = plot.process(torch_buffer(confusion));
        if (!expect(!out.has_value(), "HeatmapPlot should be a sink (no output)")) {
            return 1;
        }
        if (!expect(!view->empty() && view->snapshots().size() == 1, "view did not receive a snapshot")) {
            return 1;
        }
        const auto& snap = view->snapshots().front();
        if (!expect(snap.rows == 3 && snap.cols == 3, "snapshot dims wrong")) {
            return 1;
        }
        if (!expect(torch::allclose(snapshot_matrix(snap), torch::eye(3, torch::kFloat64)),
                    "row-normalised diagonal confusion should be identity")) {
            return 1;
        }
        if (!expect(snap.vmin == 0.0 && snap.vmax == 1.0, "normalised scale should be [0,1]")) {
            return 1;
        }
    }

    // --- No normalization: raw counts, scale [min,max] ---
    {
        auto view = std::make_shared<leakflow::plot::HeatmapView>();
        leakflow::plugins::plot::HeatmapPlot plot(view);
        plot.set_property("normalize", std::string("none"));
        (void)plot.process(torch_buffer(confusion));
        const auto& snap = view->snapshots().front();
        if (!expect(torch::allclose(snapshot_matrix(snap), confusion), "raw matrix should pass through")) {
            return 1;
        }
        if (!expect(snap.vmin == 0.0 && snap.vmax == 2.0, "raw scale should be [0,2]")) {
            return 1;
        }
    }

    // --- Batched [U,R,C] keeps the per-unit stack (no aggregation), for the unit slider ---
    {
        auto view = std::make_shared<leakflow::plot::HeatmapView>();
        leakflow::plugins::plot::HeatmapPlot plot(view);
        plot.set_property("normalize", std::string("none"));
        const auto batched = torch::stack({confusion, 3.0 * torch::eye(3, torch::kFloat64)}, 0); // [2,3,3]
        (void)plot.process(torch_buffer(batched));
        const auto& snap = view->snapshots().front();
        if (!expect(snap.units == 2, "batched snapshot should keep both units")) {
            return 1;
        }
        if (!expect(torch::allclose(snapshot_matrix(snap, 0), confusion)
                        && torch::allclose(snapshot_matrix(snap, 1), 3.0 * torch::eye(3, torch::kFloat64)),
                    "each unit matrix should be preserved")) {
            return 1;
        }
        if (!expect(snap.vmin == 0.0 && snap.vmax == 3.0, "shared scale across units should be [0,3]")) {
            return 1;
        }
    }

    // --- Pin a single unit via the `unit` property ---
    {
        auto view = std::make_shared<leakflow::plot::HeatmapView>();
        leakflow::plugins::plot::HeatmapPlot plot(view);
        plot.set_property("normalize", std::string("none"));
        plot.set_property("unit", std::int64_t{1});
        const auto batched = torch::stack({confusion, 3.0 * torch::eye(3, torch::kFloat64)}, 0);
        (void)plot.process(torch_buffer(batched));
        const auto& snap = view->snapshots().front();
        if (!expect(snap.units == 1 && torch::allclose(snapshot_matrix(snap, 0), 3.0 * torch::eye(3, torch::kFloat64)),
                    "unit=1 should pin the second unit")) {
            return 1;
        }
    }

    std::cout << "heatmap_plot tests passed\n";
    return 0;
}
