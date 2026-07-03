#include "leakflow/crypto/aes.hpp"

#include <optional>
#include <set>
#include <sstream>
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

void require_channels(std::span<const FirstRoundLeakageChannel> channels)
{
    if (channels.empty()) {
        throw std::invalid_argument("AES first-round leakage channels cannot be empty");
    }
}

void require_guess_values(const torch::Tensor& guess_values, const torch::Tensor& plaintext_blocks)
{
    require_u8_tensor(guess_values, "guess_values");
    require_same_device(guess_values, plaintext_blocks, "guess_values");
    if (guess_values.dim() != 1 || guess_values.size(0) == 0) {
        throw std::invalid_argument("guess_values must have shape [G] with G > 0");
    }
}

[[nodiscard]] std::optional<int> y_bit_index(FirstRoundLeakageChannel channel)
{
    switch (channel) {
    case FirstRoundLeakageChannel::YBit0:
        return 0;
    case FirstRoundLeakageChannel::YBit1:
        return 1;
    case FirstRoundLeakageChannel::YBit2:
        return 2;
    case FirstRoundLeakageChannel::YBit3:
        return 3;
    case FirstRoundLeakageChannel::YBit4:
        return 4;
    case FirstRoundLeakageChannel::YBit5:
        return 5;
    case FirstRoundLeakageChannel::YBit6:
        return 6;
    case FirstRoundLeakageChannel::YBit7:
        return 7;
    case FirstRoundLeakageChannel::HwM:
    case FirstRoundLeakageChannel::HwMXorK:
    case FirstRoundLeakageChannel::HwY:
        return std::nullopt;
    }

    throw std::invalid_argument("unsupported AES first-round leakage channel");
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

torch::Tensor select_y_bit(torch::Tensor y, int bit_index)
{
    const auto divisor = static_cast<std::int64_t>(1) << bit_index;
    auto shifted = torch::floor_divide(y.to(torch::kInt16), divisor);
    return torch::remainder(std::move(shifted), 2).to(torch::kUInt8).contiguous();
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

torch::Tensor known_key_leakage_channel(
    FirstRoundLeakageChannel channel,
    const torch::Tensor& plaintext_bytes,
    const torch::Tensor* key_bytes)
{
    if (channel == FirstRoundLeakageChannel::HwM) {
        return hamming_weight_u8(plaintext_bytes).contiguous();
    }

    if (key_bytes == nullptr) {
        throw std::invalid_argument("key-dependent AES first-round leakage channels require key bytes");
    }

    auto m_xor_k = torch::bitwise_xor(plaintext_bytes, *key_bytes);
    if (channel == FirstRoundLeakageChannel::HwMXorK) {
        return hamming_weight_u8(std::move(m_xor_k)).contiguous();
    }

    if (channel == FirstRoundLeakageChannel::HwY) {
        return hamming_weight_u8(sbox_u8(std::move(m_xor_k))).contiguous();
    }
    if (const auto bit = y_bit_index(channel)) {
        return select_y_bit(sbox_u8(std::move(m_xor_k)), *bit);
    }

    throw std::invalid_argument("unsupported AES first-round leakage channel");
}

torch::Tensor build_known_key_first_round_leakage(
    torch::Tensor plaintext_bytes,
    std::optional<torch::Tensor> key_bytes,
    std::span<const FirstRoundLeakageChannel> channels)
{
    require_channels(channels);

    std::vector<torch::Tensor> channel_tensors;
    channel_tensors.reserve(channels.size());
    for (const auto channel : channels) {
        channel_tensors.push_back(known_key_leakage_channel(
            channel,
            plaintext_bytes,
            key_bytes ? &*key_bytes : nullptr));
    }

    return torch::stack(channel_tensors, 2).contiguous();
}

torch::Tensor guess_domain_leakage_channel(
    FirstRoundLeakageChannel channel,
    const torch::Tensor& plaintext_bytes,
    const torch::Tensor& guess_values,
    std::optional<torch::Tensor>& m_xor_guess)
{
    const auto unit_count = plaintext_bytes.size(0);
    const auto trace_count = plaintext_bytes.size(1);
    const auto guess_count = guess_values.size(0);

    if (channel == FirstRoundLeakageChannel::HwM) {
        return hamming_weight_u8(plaintext_bytes)
            .unsqueeze(1)
            .expand({unit_count, guess_count, trace_count})
            .contiguous();
    }

    if (!m_xor_guess) {
        m_xor_guess = torch::bitwise_xor(
            plaintext_bytes.unsqueeze(1),
            guess_values.reshape({1, guess_count, 1}));
    }

    if (channel == FirstRoundLeakageChannel::HwMXorK) {
        return hamming_weight_u8(*m_xor_guess).contiguous();
    }

    if (channel == FirstRoundLeakageChannel::HwY) {
        return hamming_weight_u8(sbox_u8(*m_xor_guess)).contiguous();
    }
    if (const auto bit = y_bit_index(channel)) {
        return select_y_bit(sbox_u8(*m_xor_guess), *bit);
    }

    throw std::invalid_argument("unsupported AES first-round leakage channel");
}

} // namespace

std::string_view first_round_leakage_channel_name(
    FirstRoundLeakageChannel channel)
{
    switch (channel) {
    case FirstRoundLeakageChannel::HwM:
        return first_round_leakage_channel_hw_m;
    case FirstRoundLeakageChannel::HwMXorK:
        return first_round_leakage_channel_hw_m_xor_k;
    case FirstRoundLeakageChannel::HwY:
        return first_round_leakage_channel_hw_y;
    case FirstRoundLeakageChannel::YBit0:
        return first_round_leakage_channel_y_bits[0];
    case FirstRoundLeakageChannel::YBit1:
        return first_round_leakage_channel_y_bits[1];
    case FirstRoundLeakageChannel::YBit2:
        return first_round_leakage_channel_y_bits[2];
    case FirstRoundLeakageChannel::YBit3:
        return first_round_leakage_channel_y_bits[3];
    case FirstRoundLeakageChannel::YBit4:
        return first_round_leakage_channel_y_bits[4];
    case FirstRoundLeakageChannel::YBit5:
        return first_round_leakage_channel_y_bits[5];
    case FirstRoundLeakageChannel::YBit6:
        return first_round_leakage_channel_y_bits[6];
    case FirstRoundLeakageChannel::YBit7:
        return first_round_leakage_channel_y_bits[7];
    }

    throw std::invalid_argument("unsupported AES first-round leakage channel");
}

FirstRoundLeakageChannel first_round_leakage_channel_for(std::string_view value)
{
    if (value == first_round_leakage_channel_hw_m) {
        return FirstRoundLeakageChannel::HwM;
    }
    if (value == first_round_leakage_channel_hw_m_xor_k) {
        return FirstRoundLeakageChannel::HwMXorK;
    }
    if (value == first_round_leakage_channel_hw_y) {
        return FirstRoundLeakageChannel::HwY;
    }
    for (std::size_t index = 0; index < first_round_leakage_channel_y_bits.size(); ++index) {
        if (value == first_round_leakage_channel_y_bits[index]) {
            return static_cast<FirstRoundLeakageChannel>(
                static_cast<int>(FirstRoundLeakageChannel::YBit0) + static_cast<int>(index));
        }
    }

    throw std::invalid_argument(
        "AES first-round leakage channels must be HW(m), HW(m_xor_k), HW(y), or y(0)..y(7)");
}

std::vector<FirstRoundLeakageChannel> parse_first_round_leakage_channels(
    std::span<const std::string> values)
{
    if (values.empty()) {
        throw std::invalid_argument("AES first-round leakage channels cannot be empty");
    }

    std::set<std::string> seen;
    std::vector<FirstRoundLeakageChannel> channels;
    channels.reserve(values.size());
    for (const auto& value : values) {
        const auto channel = first_round_leakage_channel_for(value);
        const auto name = std::string(first_round_leakage_channel_name(channel));
        if (!seen.insert(name).second) {
            throw std::invalid_argument("AES first-round leakage channels must not contain duplicates");
        }
        channels.push_back(channel);
    }

    return channels;
}

bool first_round_leakage_channel_depends_on_key(
    FirstRoundLeakageChannel channel)
{
    return channel == FirstRoundLeakageChannel::HwMXorK
        || channel == FirstRoundLeakageChannel::HwY
        || y_bit_index(channel).has_value();
}

bool first_round_leakage_channels_depend_on_key(
    std::span<const FirstRoundLeakageChannel> channels)
{
    for (const auto channel : channels) {
        if (first_round_leakage_channel_depends_on_key(channel)) {
            return true;
        }
    }

    return false;
}

std::string first_round_leakage_channels_metadata(
    std::span<const FirstRoundLeakageChannel> channels)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < channels.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << first_round_leakage_channel_name(channels[index]);
    }
    return output.str();
}

std::string first_round_leakage_channel_dependencies_metadata(
    std::span<const FirstRoundLeakageChannel> channels)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < channels.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << (first_round_leakage_channel_depends_on_key(channels[index]) ? "true" : "false");
    }
    return output.str();
}

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

torch::Tensor first_round_leakage_at(
    torch::Tensor plaintext_blocks,
    std::span<const std::size_t> byte_indexes,
    std::span<const FirstRoundLeakageChannel> channels)
{
    require_plaintext_blocks(plaintext_blocks);
    if (first_round_leakage_channels_depend_on_key(channels)) {
        throw std::invalid_argument("key-dependent AES first-round leakage channels require key bytes");
    }
    return build_known_key_first_round_leakage(
        select_byte_columns(plaintext_blocks, byte_indexes),
        std::nullopt,
        channels);
}

torch::Tensor first_round_leakage_at(
    torch::Tensor key_byte_or_blocks,
    torch::Tensor plaintext_blocks,
    std::span<const std::size_t> byte_indexes,
    std::span<const FirstRoundLeakageChannel> channels)
{
    require_plaintext_blocks(plaintext_blocks);
    auto plaintext_bytes = select_byte_columns(plaintext_blocks, byte_indexes);

    std::optional<torch::Tensor> key_bytes;
    if (first_round_leakage_channels_depend_on_key(channels)) {
        key_bytes = normalize_key_bytes_for_block_indexes(
            std::move(key_byte_or_blocks),
            plaintext_blocks,
            byte_indexes);
    }

    return build_known_key_first_round_leakage(
        std::move(plaintext_bytes),
        std::move(key_bytes),
        channels);
}

torch::Tensor first_round_leakage_hypotheses_at(
    torch::Tensor guess_values,
    torch::Tensor plaintext_blocks,
    std::span<const std::size_t> byte_indexes,
    std::span<const FirstRoundLeakageChannel> channels)
{
    require_plaintext_blocks(plaintext_blocks);
    require_guess_values(guess_values, plaintext_blocks);
    require_channels(channels);

    auto plaintext_bytes = select_byte_columns(plaintext_blocks, byte_indexes);
    std::optional<torch::Tensor> m_xor_guess;
    std::vector<torch::Tensor> channel_tensors;
    channel_tensors.reserve(channels.size());
    for (const auto channel : channels) {
        channel_tensors.push_back(guess_domain_leakage_channel(
            channel,
            plaintext_bytes,
            guess_values,
            m_xor_guess));
    }

    return torch::stack(channel_tensors, 3).contiguous();
}

} // namespace leakflow::crypto::aes
