#include "leakflow/crypto/hamming.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow::crypto {
namespace {

void require_u8_tensor(const torch::Tensor& tensor, std::string_view name)
{
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be a defined uint8 Torch tensor");
    }
    if (tensor.dtype() != torch::kUInt8) {
        throw std::invalid_argument(std::string(name) + " must have dtype uint8");
    }
    if (tensor.layout() != torch::kStrided) {
        throw std::invalid_argument(std::string(name) + " must use a strided Torch layout");
    }
}

const torch::Tensor& cpu_hamming_weight_lut()
{
    static const auto lut = torch::from_blob(
                                const_cast<Byte*>(hamming_weight_table.data()),
                                {static_cast<std::int64_t>(hamming_weight_table.size())},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                .clone();
    return lut;
}

torch::Tensor reshape_like(torch::Tensor tensor, const torch::Tensor& reference)
{
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(reference.dim()));
    for (const auto dimension : reference.sizes()) {
        shape.push_back(dimension);
    }
    return tensor.reshape(shape);
}

torch::Tensor lookup_hamming_weight(torch::Tensor values)
{
    auto lut = cpu_hamming_weight_lut();
    if (!values.device().is_cpu()) {
        lut = lut.to(values.device());
    }

    auto flat_indices = values.to(torch::kLong).reshape({-1});
    auto looked_up = lut.index_select(0, flat_indices);
    return reshape_like(std::move(looked_up), values);
}

} // namespace

torch::Tensor hamming_weight_u8(torch::Tensor values)
{
    require_u8_tensor(values, "hamming_weight_u8 values");
    return lookup_hamming_weight(std::move(values));
}

torch::Tensor hamming_distance_u8(torch::Tensor lhs, torch::Tensor rhs)
{
    require_u8_tensor(lhs, "hamming_distance_u8 lhs");
    require_u8_tensor(rhs, "hamming_distance_u8 rhs");
    if (lhs.device() != rhs.device()) {
        throw std::invalid_argument("hamming_distance_u8 tensors must be on the same device");
    }

    return hamming_weight_u8(torch::bitwise_xor(lhs, rhs));
}

} // namespace leakflow::crypto
