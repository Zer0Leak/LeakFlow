#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <torch/torch.h>

namespace leakflow::crypto {

using Byte = std::uint8_t;

namespace detail {

[[nodiscard]] constexpr Byte hamming_weight_byte(Byte value) noexcept
{
    Byte count = 0;
    while (value != 0) {
        count = static_cast<Byte>(count + (value & Byte{1}));
        value = static_cast<Byte>(value >> 1U);
    }
    return count;
}

[[nodiscard]] constexpr std::array<Byte, 256> make_hamming_weight_table() noexcept
{
    std::array<Byte, 256> table{};
    for (std::size_t index = 0; index < table.size(); ++index) {
        table[index] = hamming_weight_byte(static_cast<Byte>(index));
    }
    return table;
}

} // namespace detail

inline constexpr auto hamming_weight_table = detail::make_hamming_weight_table();

[[nodiscard]] constexpr Byte hamming_weight(Byte value) noexcept
{
    return hamming_weight_table[static_cast<std::size_t>(value)];
}

[[nodiscard]] constexpr unsigned hamming_weight(std::uint16_t value) noexcept
{
    return hamming_weight(static_cast<Byte>(value & std::uint16_t{0x00ff}))
        + hamming_weight(static_cast<Byte>((value >> 8U) & std::uint16_t{0x00ff}));
}

[[nodiscard]] constexpr unsigned hamming_weight(std::uint32_t value) noexcept
{
    return hamming_weight(static_cast<Byte>(value & std::uint32_t{0x000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 8U) & std::uint32_t{0x000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 16U) & std::uint32_t{0x000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 24U) & std::uint32_t{0x000000ff}));
}

[[nodiscard]] constexpr unsigned hamming_weight(std::uint64_t value) noexcept
{
    return hamming_weight(static_cast<Byte>(value & std::uint64_t{0x00000000000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 8U) & std::uint64_t{0x00000000000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 16U) & std::uint64_t{0x00000000000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 24U) & std::uint64_t{0x00000000000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 32U) & std::uint64_t{0x00000000000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 40U) & std::uint64_t{0x00000000000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 48U) & std::uint64_t{0x00000000000000ff}))
        + hamming_weight(static_cast<Byte>((value >> 56U) & std::uint64_t{0x00000000000000ff}));
}

[[nodiscard]] constexpr Byte hamming_distance(Byte lhs, Byte rhs) noexcept
{
    return hamming_weight(static_cast<Byte>(lhs ^ rhs));
}

[[nodiscard]] constexpr unsigned hamming_distance(std::uint16_t lhs, std::uint16_t rhs) noexcept
{
    return hamming_weight(static_cast<std::uint16_t>(lhs ^ rhs));
}

[[nodiscard]] constexpr unsigned hamming_distance(std::uint32_t lhs, std::uint32_t rhs) noexcept
{
    return hamming_weight(static_cast<std::uint32_t>(lhs ^ rhs));
}

[[nodiscard]] constexpr unsigned hamming_distance(std::uint64_t lhs, std::uint64_t rhs) noexcept
{
    return hamming_weight(static_cast<std::uint64_t>(lhs ^ rhs));
}

[[nodiscard]] torch::Tensor hamming_weight_u8(torch::Tensor values);
[[nodiscard]] torch::Tensor hamming_distance_u8(torch::Tensor lhs, torch::Tensor rhs);

} // namespace leakflow::crypto
