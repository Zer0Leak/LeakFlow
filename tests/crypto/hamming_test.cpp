#include "leakflow/crypto/hamming.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

template <typename Exception, typename Function>
bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }

    return false;
}

} // namespace

int main()
{
    using leakflow::crypto::Byte;

    static_assert(leakflow::crypto::hamming_weight(Byte{0x00}) == 0);
    static_assert(leakflow::crypto::hamming_weight(Byte{0xff}) == 8);
    static_assert(leakflow::crypto::hamming_weight(Byte{0x53}) == 4);
    static_assert(leakflow::crypto::hamming_weight(std::uint16_t{0xffff}) == 16);
    static_assert(leakflow::crypto::hamming_weight(std::uint32_t{0xffffffff}) == 32);
    static_assert(leakflow::crypto::hamming_weight(std::uint64_t{0xffffffffffffffffULL}) == 64);
    static_assert(leakflow::crypto::hamming_distance(Byte{0x00}, Byte{0xff}) == 8);
    static_assert(leakflow::crypto::hamming_distance(std::uint16_t{0x0f0f}, std::uint16_t{0xf0f0}) == 16);

    const auto options = torch::TensorOptions().dtype(torch::kUInt8);
    auto values = torch::tensor({0x00, 0x01, 0x7f, 0x80, 0xff}, options);
    auto weights = leakflow::crypto::hamming_weight_u8(values);

    if (!expect(weights.dtype() == torch::kUInt8, "hamming_weight_u8 dtype changed")) {
        return 1;
    }
    if (!expect(weights.dim() == 1 && weights.size(0) == 5, "hamming_weight_u8 shape mismatch")) {
        return 1;
    }

    const auto* weight_data = weights.data_ptr<Byte>();
    const Byte expected_weights[]{0, 1, 7, 1, 8};
    for (std::size_t index = 0; index < 5; ++index) {
        if (!expect(weight_data[index] == expected_weights[index], "hamming_weight_u8 value mismatch")) {
            return 1;
        }
    }

    auto scalar = torch::tensor(0xff, options);
    auto scalar_weight = leakflow::crypto::hamming_weight_u8(scalar);
    if (!expect(scalar_weight.dim() == 0, "scalar hamming_weight_u8 should return a scalar")) {
        return 1;
    }
    if (!expect(scalar_weight.item<Byte>() == 8, "scalar hamming_weight_u8 value mismatch")) {
        return 1;
    }

    auto lhs = torch::tensor({0x00, 0x0f, 0xaa}, options);
    auto rhs = torch::tensor({0xff, 0xf0, 0x55}, options);
    auto distances = leakflow::crypto::hamming_distance_u8(lhs, rhs);
    const auto* distance_data = distances.data_ptr<Byte>();
    for (std::size_t index = 0; index < 3; ++index) {
        if (!expect(distance_data[index] == 8, "hamming_distance_u8 value mismatch")) {
            return 1;
        }
    }

    if (!expect(throws_exception<std::invalid_argument>([] {
            (void)leakflow::crypto::hamming_weight_u8(torch::tensor({1, 2, 3}, torch::kInt32));
        }),
            "hamming_weight_u8 should reject non-uint8 tensors")) {
        return 1;
    }

    return 0;
}
