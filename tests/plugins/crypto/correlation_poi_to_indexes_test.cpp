#include "leakflow/plugins/crypto/correlation_poi_to_indexes.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <torch/torch.h>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

// Build a [channel, k, 2] result: [...,0] = sample indexes, [...,1] = scores.
torch::Tensor make_result(torch::Tensor indexes)
{
    return torch::stack({indexes.to(torch::kFloat64), torch::zeros_like(indexes.to(torch::kFloat64))}, 2);
}

} // namespace

int main()
{
    std::vector<leakflow::plugins::crypto::CorrelationPoiResult> results;
    results.push_back({0, make_result(torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kLong))});
    results.push_back({1, make_result(torch::tensor({{30, 31, 32}, {40, 41, 42}}, torch::kLong))});
    auto payload = std::make_shared<leakflow::plugins::crypto::CorrelationPoiPayload>(
        std::move(results), std::string("correlation"));

    leakflow::Buffer buffer{leakflow::Caps(leakflow::plugins::crypto::correlation_poi_caps_type)};
    buffer.set_payload(payload);

    leakflow::plugins::crypto::CorrelationPoiToIndexes element;
    const auto out = element.process(std::move(buffer));
    if (!expect(out.has_value(), "no output")) {
        return 1;
    }
    const auto indexes = out->payload_as<leakflow::base::TorchTensorPayload>()->tensor();
    // Per byte, the two channels' indexes are concatenated: [10,11,12, 20,21,22] etc.
    const auto expected = torch::tensor({{10, 11, 12, 20, 21, 22}, {30, 31, 32, 40, 41, 42}}, torch::kLong);
    if (!expect(indexes.sizes() == torch::IntArrayRef({2, 6}), "index shape wrong (want [U, channels*k])")) {
        return 1;
    }
    if (!expect(torch::equal(indexes, expected), "index values wrong")) {
        return 1;
    }
    if (!expect(out->metadata_or("payload.feature.selected_count", "") == "6", "selected_count metadata wrong")) {
        return 1;
    }

    // --- units filter: keep only byte unit 1 -> [1, 6] ---
    {
        std::vector<leakflow::plugins::crypto::CorrelationPoiResult> r2;
        r2.push_back({0, make_result(torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kLong))});
        r2.push_back({1, make_result(torch::tensor({{30, 31, 32}, {40, 41, 42}}, torch::kLong))});
        auto p2 = std::make_shared<leakflow::plugins::crypto::CorrelationPoiPayload>(
            std::move(r2), std::string("correlation"));
        leakflow::Buffer b2{leakflow::Caps(leakflow::plugins::crypto::correlation_poi_caps_type)};
        b2.set_payload(p2);
        leakflow::plugins::crypto::CorrelationPoiToIndexes filtered;
        filtered.set_property("units", leakflow::IntList{1});
        const auto out2 = filtered.process(std::move(b2));
        const auto idx2 = out2->payload_as<leakflow::base::TorchTensorPayload>()->tensor();
        const auto want = torch::tensor({{30, 31, 32, 40, 41, 42}}, torch::kLong);
        if (!expect(idx2.sizes() == torch::IntArrayRef({1, 6}) && torch::equal(idx2, want),
                    "units filter did not select byte unit 1")) {
            return 1;
        }
    }

    std::cout << "correlation_poi_to_indexes tests passed\n";
    return 0;
}
