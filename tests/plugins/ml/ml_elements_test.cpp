#include "leakflow/plugins/ml/clustering_stats.hpp"
#include "leakflow/plugins/ml/feature_select.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/buffer.hpp"

#include <iostream>
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

leakflow::ElementInputs two_inputs(const char* a, torch::Tensor ta, const char* b, torch::Tensor tb)
{
    leakflow::ElementInputs inputs;
    inputs.emplace(a, torch_buffer(std::move(ta)));
    inputs.emplace(b, torch_buffer(std::move(tb)));
    return inputs;
}

} // namespace

int main()
{
    // --- FeatureSelect: 1-D shared indexes ---
    {
        const auto features = torch::arange(30, torch::kFloat64).reshape({6, 5}); // [6,5]
        const auto indexes = torch::tensor({0, 2, 4}, torch::kLong);
        leakflow::plugins::ml::FeatureSelect element;
        const auto out = element.process_inputs(two_inputs("features", features, "indexes", indexes));
        if (!expect(out.has_value(), "FeatureSelect 1-D: no output")) {
            return 1;
        }
        const auto selected = out->payload_as<leakflow::base::TorchTensorPayload>()->tensor();
        if (!expect(selected.sizes() == torch::IntArrayRef({6, 3}), "FeatureSelect 1-D: wrong shape")) {
            return 1;
        }
        if (!expect(torch::equal(selected, features.index_select(1, indexes)), "FeatureSelect 1-D: wrong values")) {
            return 1;
        }
        if (!expect(out->metadata_or("payload.layout", "") == "observation/feature",
                "FeatureSelect 1-D: wrong payload layout")) {
            return 1;
        }
    }

    // --- FeatureSelect: 2-D per-unit indexes broadcast a shared [T,N] matrix ---
    {
        const auto features = torch::arange(30, torch::kFloat64).reshape({6, 5}); // [6,5]
        const auto indexes = torch::tensor({{0, 1}, {3, 4}}, torch::kLong);       // [2,2]
        leakflow::plugins::ml::FeatureSelect element;
        const auto out = element.process_inputs(two_inputs("features", features, "indexes", indexes));
        const auto selected = out->payload_as<leakflow::base::TorchTensorPayload>()->tensor();
        if (!expect(selected.sizes() == torch::IntArrayRef({2, 6, 2}), "FeatureSelect 2-D: wrong shape")) {
            return 1;
        }
        if (!expect(torch::equal(selected[0], features.index_select(1, indexes[0]))
                        && torch::equal(selected[1], features.index_select(1, indexes[1])),
                    "FeatureSelect 2-D: wrong per-unit values")) {
            return 1;
        }
        if (!expect(out->metadata_or("payload.layout", "") == "unit/observation/feature",
                "FeatureSelect 2-D: wrong payload layout")) {
            return 1;
        }
    }

    // --- ClusteringStats: perfect (permuted) clustering => diagonal confusion, accuracy 1 ---
    {
        const auto labels = torch::tensor({0, 0, 1, 1, 2, 2}, torch::kLong);
        const auto truth = torch::tensor({2, 2, 0, 0, 1, 1}, torch::kLong); // relabelled but identical partition
        leakflow::plugins::ml::ClusteringStats element;
        const auto out = element.process_inputs(two_inputs("labels", labels, "truth", truth));
        if (!expect(out.has_value(), "ClusteringStats: no output")) {
            return 1;
        }
        const auto reordered = out->payload_as<leakflow::base::TorchTensorPayload>()->tensor();
        if (!expect(torch::equal(reordered, 2.0 * torch::eye(3, torch::kFloat64)),
                    "ClusteringStats: reordered confusion not diagonal")) {
            return 1;
        }
        if (!expect(out->metadata_or("payload.cluster_stats.accuracy", "") == "1.000000", "accuracy metadata wrong")) {
            return 1;
        }
        if (!expect(out->metadata_or("payload.cluster_stats.ari", "") == "1.000000", "ari metadata wrong")) {
            return 1;
        }
        if (!expect(out->metadata_or("payload.layout", "") == "true_class/cluster",
                "ClusteringStats payload layout wrong")) {
            return 1;
        }
    }

    std::cout << "ml_elements tests passed\n";
    return 0;
}
