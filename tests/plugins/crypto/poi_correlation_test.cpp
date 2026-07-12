#include "leakflow/plugins/crypto/poi_correlation.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

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

leakflow::Buffer torch_buffer(torch::Tensor tensor)
{
    auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    leakflow::Buffer buffer{payload.caps()};
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    return buffer;
}

} // namespace

int main()
{
    torch::manual_seed(3);
    const auto target = torch::randn({100}, torch::kFloat64);
    auto traces = torch::randn({100, 10}, torch::kFloat64);
    traces.select(1, 3).copy_(target);        // column 3 == target      -> corr +1
    traces.select(1, 7).copy_(-1.0 * target); // column 7 == -target     -> corr -1
    const auto leakage = target.reshape({1, 100, 1}); // [B=1, N=100, C=1]

    // One byte, one channel, two PoIs at samples 3 and 7 (dummy original scores).
    std::vector<leakflow::plugins::crypto::CorrelationPoiResult> results;
    results.push_back({0, torch::tensor({{{3.0, 0.5}, {7.0, 0.5}}}, torch::kFloat64)}); // [1,2,2]
    auto poi = std::make_shared<leakflow::plugins::crypto::CorrelationPoiPayload>(
        std::move(results), std::string("correlation"));
    leakflow::Buffer poi_buffer{leakflow::Caps(leakflow::plugins::crypto::correlation_poi_caps_type)};
    poi_buffer.set_payload(poi);

    leakflow::ElementInputs inputs;
    inputs.emplace("poi", std::move(poi_buffer));
    inputs.emplace("traces", torch_buffer(traces));
    auto target_buffer = torch_buffer(leakage);
    target_buffer.set_metadata("attack.unit.indexes", "[0]");
    target_buffer.set_metadata("attack.unit.count", "1");
    inputs.emplace("targets", std::move(target_buffer));

    leakflow::plugins::crypto::PoiCorrelation element;
    const auto out = element.process_inputs(std::move(inputs));
    if (!expect(out.has_value(), "no output")) {
        return 1;
    }
    const auto rescored = out->payload_as<leakflow::plugins::crypto::CorrelationPoiPayload>();
    if (!expect(rescored != nullptr && rescored->results().size() == 1, "output is not a rescored PoI payload")) {
        return 1;
    }
    if (!expect(out->metadata("payload.layout") == "unit/channel/poi/[sample_index,correlation]"
                    && out->metadata("payload.poi.unit_count") == "1"
                    && out->metadata("attack.unit.indexes") == "[0]",
                "rescored PoI unit/layout metadata was wrong")) {
        return 1;
    }
    const auto& r = rescored->results().front().result; // [1,2,2]
    // Positions preserved; scores replaced by the correlation on the new traces (+1 and -1).
    if (!expect(r[0][0][0].item<double>() == 3.0 && r[0][1][0].item<double>() == 7.0, "positions not preserved")) {
        return 1;
    }
    if (!expect(std::abs(r[0][0][1].item<double>() - 1.0) < 1.0e-6, "PoI at sample 3 should re-score to +1")) {
        return 1;
    }
    if (!expect(std::abs(r[0][1][1].item<double>() + 1.0) < 1.0e-6, "PoI at sample 7 should re-score to -1")) {
        return 1;
    }

    std::cout << "poi_correlation tests passed\n";
    return 0;
}
