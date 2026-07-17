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
    if (!expect(out->metadata_or("payload.layout", "") == "unit/feature"
                    && out->metadata_or("attack.unit.indexes", "") == "[0,1]"
                    && out->metadata_or("attack.unit.count", "") == "2",
                "index payload unit/layout metadata wrong")) {
        return 1;
    }

    std::cout << "correlation_poi_to_indexes tests passed\n";
    return 0;
}
