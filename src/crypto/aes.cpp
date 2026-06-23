#include "leakflow/crypto/aes.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow::crypto::aes {
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

void require_same_device(const torch::Tensor& lhs, const torch::Tensor& rhs, std::string_view name)
{
    if (lhs.device() != rhs.device()) {
        throw std::invalid_argument(std::string(name) + " must be on the same device as plaintext");
    }
}

void require_plaintext_bytes(const torch::Tensor& plaintext_bytes)
{
    require_u8_tensor(plaintext_bytes, "plaintext_bytes");
    if (plaintext_bytes.dim() != 1) {
        throw std::invalid_argument("plaintext_bytes must have shape [N]");
    }
}

void require_plaintext_blocks(const torch::Tensor& plaintext_blocks)
{
    require_u8_tensor(plaintext_blocks, "plaintext_blocks");
    if (plaintext_blocks.dim() != 2 || plaintext_blocks.size(1) != 16) {
        throw std::invalid_argument("plaintext_blocks must have shape [N,16]");
    }
}

void require_byte_index(std::size_t byte_index)
{
    if (byte_index >= 16) {
        throw std::invalid_argument("AES byte_index must be in the range [0,15]");
    }
}

const torch::Tensor& cpu_sbox_lut()
{
    static const auto lut = torch::from_blob(
                                const_cast<Byte*>(sbox_table.data()),
                                {static_cast<std::int64_t>(sbox_table.size())},
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

torch::Tensor sbox_u8(torch::Tensor values)
{
    auto lut = cpu_sbox_lut();
    if (!values.device().is_cpu()) {
        lut = lut.to(values.device());
    }

    auto flat_indices = values.to(torch::kLong).reshape({-1});
    auto looked_up = lut.index_select(0, flat_indices);
    return reshape_like(std::move(looked_up), values);
}

torch::Tensor broadcast_key_byte(Byte key_byte, const torch::Tensor& plaintext_bytes)
{
    return torch::full(
        plaintext_bytes.sizes(),
        static_cast<std::int64_t>(key_byte),
        plaintext_bytes.options());
}

torch::Tensor normalize_key_bytes(torch::Tensor key_byte_or_bytes, const torch::Tensor& plaintext_bytes)
{
    require_u8_tensor(key_byte_or_bytes, "key_byte_or_bytes");
    require_same_device(key_byte_or_bytes, plaintext_bytes, "key_byte_or_bytes");

    const auto trace_count = plaintext_bytes.size(0);
    if (key_byte_or_bytes.dim() == 0) {
        return key_byte_or_bytes.reshape({1}).expand({trace_count});
    }
    if (key_byte_or_bytes.dim() == 1) {
        if (key_byte_or_bytes.size(0) == 1) {
            return key_byte_or_bytes.expand({trace_count});
        }
        if (key_byte_or_bytes.size(0) == trace_count) {
            return key_byte_or_bytes;
        }
    }

    throw std::invalid_argument("key_byte_or_bytes must be a scalar, [1], or [N] uint8 tensor");
}

torch::Tensor select_byte_column(const torch::Tensor& blocks, std::size_t byte_index)
{
    return blocks.index({torch::indexing::Slice(), static_cast<std::int64_t>(byte_index)});
}

torch::Tensor byte_index_tensor(std::span<const std::size_t> byte_indexes, const torch::Device& device)
{
    std::vector<std::int64_t> indexes;
    indexes.reserve(byte_indexes.size());
    for (const auto byte_index : byte_indexes) {
        require_byte_index(byte_index);
        indexes.push_back(static_cast<std::int64_t>(byte_index));
    }

    return torch::tensor(indexes, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU)).to(device);
}

torch::Tensor select_byte_columns(const torch::Tensor& blocks, std::span<const std::size_t> byte_indexes)
{
    return blocks.index_select(1, byte_index_tensor(byte_indexes, blocks.device())).transpose(0, 1).contiguous();
}

torch::Tensor normalize_key_bytes_for_blocks(
    torch::Tensor key_byte_or_blocks,
    const torch::Tensor& plaintext_blocks,
    std::size_t byte_index)
{
    require_u8_tensor(key_byte_or_blocks, "key_byte_or_blocks");
    require_same_device(key_byte_or_blocks, plaintext_blocks, "key_byte_or_blocks");

    const auto trace_count = plaintext_blocks.size(0);
    if (key_byte_or_blocks.dim() == 0) {
        return key_byte_or_blocks.reshape({1}).expand({trace_count});
    }
    if (key_byte_or_blocks.dim() == 1) {
        if (key_byte_or_blocks.size(0) == 1) {
            return key_byte_or_blocks.expand({trace_count});
        }
        if (key_byte_or_blocks.size(0) == trace_count) {
            return key_byte_or_blocks;
        }
    }
    if (key_byte_or_blocks.dim() == 2) {
        if (key_byte_or_blocks.size(0) == trace_count && key_byte_or_blocks.size(1) == 16) {
            return select_byte_column(key_byte_or_blocks, byte_index);
        }
    }

    throw std::invalid_argument("key_byte_or_blocks must be a scalar, [1], [N], or [N,16] uint8 tensor");
}

torch::Tensor normalize_key_bytes_for_block_indexes(
    torch::Tensor key_byte_or_blocks,
    const torch::Tensor& plaintext_blocks,
    std::span<const std::size_t> byte_indexes)
{
    require_u8_tensor(key_byte_or_blocks, "key_byte_or_blocks");
    require_same_device(key_byte_or_blocks, plaintext_blocks, "key_byte_or_blocks");

    const auto trace_count = plaintext_blocks.size(0);
    const auto byte_count = static_cast<std::int64_t>(byte_indexes.size());
    if (key_byte_or_blocks.dim() == 0) {
        return key_byte_or_blocks.reshape({1, 1}).expand({byte_count, trace_count});
    }
    if (key_byte_or_blocks.dim() == 1) {
        if (key_byte_or_blocks.size(0) == 1) {
            return key_byte_or_blocks.reshape({1, 1}).expand({byte_count, trace_count});
        }
        if (key_byte_or_blocks.size(0) == trace_count) {
            return key_byte_or_blocks.reshape({1, trace_count}).expand({byte_count, trace_count});
        }
    }
    if (key_byte_or_blocks.dim() == 2) {
        if (key_byte_or_blocks.size(0) == trace_count && key_byte_or_blocks.size(1) == 16) {
            return select_byte_columns(key_byte_or_blocks, byte_indexes);
        }
    }

    throw std::invalid_argument("key_byte_or_blocks must be a scalar, [1], [N], or [N,16] uint8 tensor");
}

FirstRoundSboxLeakageTensors build_first_round_sbox_leakage(
    torch::Tensor key_bytes,
    torch::Tensor plaintext_bytes)
{
    auto y = sbox_u8(torch::bitwise_xor(plaintext_bytes, key_bytes));
    const auto output_axis = plaintext_bytes.dim();
    auto values = torch::stack({plaintext_bytes, y}, output_axis).contiguous();
    auto hamming_weights = torch::stack({hamming_weight_u8(plaintext_bytes), hamming_weight_u8(y)}, output_axis)
                               .contiguous();

    return FirstRoundSboxLeakageTensors{
        .values = std::move(values),
        .hamming_weights = std::move(hamming_weights),
    };
}

} // namespace

FirstRoundSboxLeakageTensors first_round_sbox_leakage(
    Byte key_byte,
    torch::Tensor plaintext_bytes)
{
    require_plaintext_bytes(plaintext_bytes);
    return build_first_round_sbox_leakage(
        broadcast_key_byte(key_byte, plaintext_bytes),
        std::move(plaintext_bytes));
}

FirstRoundSboxLeakageTensors first_round_sbox_leakage(
    torch::Tensor key_byte_or_bytes,
    torch::Tensor plaintext_bytes)
{
    require_plaintext_bytes(plaintext_bytes);
    auto key_bytes = normalize_key_bytes(std::move(key_byte_or_bytes), plaintext_bytes);
    return build_first_round_sbox_leakage(std::move(key_bytes), std::move(plaintext_bytes));
}

FirstRoundSboxLeakageTensors first_round_sbox_leakage_at(
    Byte key_byte,
    torch::Tensor plaintext_blocks,
    std::size_t byte_index)
{
    require_byte_index(byte_index);
    require_plaintext_blocks(plaintext_blocks);
    return first_round_sbox_leakage(key_byte, select_byte_column(plaintext_blocks, byte_index));
}

FirstRoundSboxLeakageTensors first_round_sbox_leakage_at(
    Byte key_byte,
    torch::Tensor plaintext_blocks,
    std::span<const std::size_t> byte_indexes)
{
    require_plaintext_blocks(plaintext_blocks);
    auto plaintext_bytes = select_byte_columns(plaintext_blocks, byte_indexes);
    auto key_bytes = torch::full(
        plaintext_bytes.sizes(),
        static_cast<std::int64_t>(key_byte),
        plaintext_bytes.options());
    return build_first_round_sbox_leakage(std::move(key_bytes), std::move(plaintext_bytes));
}

FirstRoundSboxLeakageTensors first_round_sbox_leakage_at(
    torch::Tensor key_byte_or_blocks,
    torch::Tensor plaintext_blocks,
    std::size_t byte_index)
{
    require_byte_index(byte_index);
    require_plaintext_blocks(plaintext_blocks);
    auto key_bytes = normalize_key_bytes_for_blocks(std::move(key_byte_or_blocks), plaintext_blocks, byte_index);
    return build_first_round_sbox_leakage(
        std::move(key_bytes),
        select_byte_column(plaintext_blocks, byte_index));
}

FirstRoundSboxLeakageTensors first_round_sbox_leakage_at(
    torch::Tensor key_byte_or_blocks,
    torch::Tensor plaintext_blocks,
    std::span<const std::size_t> byte_indexes)
{
    require_plaintext_blocks(plaintext_blocks);
    auto plaintext_bytes = select_byte_columns(plaintext_blocks, byte_indexes);
    auto key_bytes = normalize_key_bytes_for_block_indexes(std::move(key_byte_or_blocks), plaintext_blocks, byte_indexes);
    return build_first_round_sbox_leakage(std::move(key_bytes), std::move(plaintext_bytes));
}

} // namespace leakflow::crypto::aes
