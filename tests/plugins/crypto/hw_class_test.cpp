#include "leakflow/plugins/crypto/hw_class.hpp"

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

} // namespace

int main()
{
    // Leakage [B=2, N=3, C=2]: channel 0 = HW(m), channel 1 = HW(y). class = 9*HW(m) + HW(y).
    const auto hw_m = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kUInt8);
    const auto hw_y = torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kUInt8);
    const auto leakage = torch::stack({hw_m, hw_y}, 2); // [2,3,2]

    leakflow::plugins::crypto::HwClass element;
    const auto out = element.process(torch_buffer(leakage));
    if (!expect(out.has_value(), "HwClass: no output")) {
        return 1;
    }
    const auto classes = out->payload_as<leakflow::base::TorchTensorPayload>()->tensor();
    const auto expected = torch::tensor({{9, 19, 29}, {39, 49, 59}}, torch::kLong);
    if (!expect(classes.sizes() == torch::IntArrayRef({2, 3}), "HwClass: wrong shape")) {
        return 1;
    }
    if (!expect(torch::equal(classes, expected), "HwClass: wrong class values")) {
        return 1;
    }
    if (!expect(out->metadata_or("payload.class.count", "") == "81", "HwClass: class count metadata wrong")) {
        return 1;
    }
    if (!expect(out->metadata_or("payload.layout", "") == "unit/trace", "HwClass: payload layout wrong")) {
        return 1;
    }

    std::cout << "hw_class tests passed\n";
    return 0;
}
