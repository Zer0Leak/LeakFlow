#include "leakflow/crypto/aes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <vector>

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

bool expect_tensor_row(
    const torch::Tensor& values,
    const torch::Tensor& hamming_weights,
    std::int64_t row,
    leakflow::crypto::Byte key_byte,
    leakflow::crypto::Byte plaintext_byte)
{
    const auto expected = leakflow::crypto::aes::first_round_sbox_leakage(key_byte, plaintext_byte);
    const auto* value_data = values.data_ptr<leakflow::crypto::Byte>();
    const auto* hamming_data = hamming_weights.data_ptr<leakflow::crypto::Byte>();
    const auto offset = static_cast<std::size_t>(row * 2);

    return expect(value_data[offset] == expected.m, "leakage tensor m mismatch")
        && expect(value_data[offset + 1] == expected.y, "leakage tensor y mismatch")
        && expect(hamming_data[offset] == expected.hw_m, "leakage tensor HW(m) mismatch")
        && expect(hamming_data[offset + 1] == expected.hw_y, "leakage tensor HW(y) mismatch");
}

bool expect_tensor_plane_row(
    const torch::Tensor& values,
    const torch::Tensor& hamming_weights,
    std::int64_t plane,
    std::int64_t row,
    leakflow::crypto::Byte key_byte,
    leakflow::crypto::Byte plaintext_byte)
{
    const auto expected = leakflow::crypto::aes::first_round_sbox_leakage(key_byte, plaintext_byte);
    const auto* value_data = values.data_ptr<leakflow::crypto::Byte>();
    const auto* hamming_data = hamming_weights.data_ptr<leakflow::crypto::Byte>();
    const auto row_count = values.size(1);
    const auto offset = static_cast<std::size_t>(((plane * row_count) + row) * 2);

    return expect(value_data[offset] == expected.m, "multi-index leakage tensor m mismatch")
        && expect(value_data[offset + 1] == expected.y, "multi-index leakage tensor y mismatch")
        && expect(hamming_data[offset] == expected.hw_m, "multi-index leakage tensor HW(m) mismatch")
        && expect(hamming_data[offset + 1] == expected.hw_y, "multi-index leakage tensor HW(y) mismatch");
}

bool expect_leakage_channel(
    const torch::Tensor& leakage,
    std::int64_t unit,
    std::int64_t trace,
    std::int64_t channel,
    leakflow::crypto::Byte expected,
    const char* message)
{
    const auto trace_count = leakage.size(1);
    const auto channel_count = leakage.size(2);
    const auto* data = leakage.data_ptr<leakflow::crypto::Byte>();
    const auto offset = static_cast<std::size_t>(((unit * trace_count) + trace) * channel_count + channel);
    return expect(data[offset] == expected, message);
}

bool expect_hypothesis_channel(
    const torch::Tensor& hypotheses,
    std::int64_t unit,
    std::int64_t guess,
    std::int64_t trace,
    std::int64_t channel,
    leakflow::crypto::Byte expected,
    const char* message)
{
    const auto guess_count = hypotheses.size(1);
    const auto trace_count = hypotheses.size(2);
    const auto channel_count = hypotheses.size(3);
    const auto* data = hypotheses.data_ptr<leakflow::crypto::Byte>();
    const auto offset = static_cast<std::size_t>(
        (((unit * guess_count) + guess) * trace_count + trace) * channel_count + channel);
    return expect(data[offset] == expected, message);
}

} // namespace

int main()
{
    using leakflow::crypto::Byte;

    static_assert(leakflow::crypto::aes::sbox(Byte{0x00}) == Byte{0x63});
    static_assert(leakflow::crypto::aes::sbox(Byte{0x53}) == Byte{0xed});
    static_assert(leakflow::crypto::aes::first_round_sbox_output(Byte{0x2b}, Byte{0x32}) == Byte{0xd4});
    static_assert(leakflow::crypto::aes::first_round_sbox_output_hw(Byte{0x2b}, Byte{0x32}) == Byte{4});
    static_assert(leakflow::crypto::aes::first_round_sbox_leakage(Byte{0x2b}, Byte{0x32}).m == Byte{0x32});
    static_assert(leakflow::crypto::aes::first_round_sbox_leakage(Byte{0x2b}, Byte{0x32}).y == Byte{0xd4});
    static_assert(leakflow::crypto::aes::first_round_sbox_leakage(Byte{0x2b}, Byte{0x32}).hw_m == Byte{3});
    static_assert(leakflow::crypto::aes::first_round_sbox_leakage(Byte{0x2b}, Byte{0x32}).hw_y == Byte{4});

    const auto options = torch::TensorOptions().dtype(torch::kUInt8);
    auto plaintext_bytes = torch::tensor({0x32, 0x00, 0xff}, options);

    auto scalar_key_result = leakflow::crypto::aes::first_round_sbox_leakage(Byte{0x2b}, plaintext_bytes);
    if (!expect(scalar_key_result.values.dtype() == torch::kUInt8, "leakage values dtype changed")) {
        return 1;
    }
    if (!expect(scalar_key_result.hamming_weights.dtype() == torch::kUInt8, "leakage HW dtype changed")) {
        return 1;
    }
    if (!expect(scalar_key_result.values.dim() == 2 && scalar_key_result.values.size(0) == 3
            && scalar_key_result.values.size(1) == 2,
            "leakage values shape mismatch")) {
        return 1;
    }
    if (!expect(scalar_key_result.hamming_weights.dim() == 2 && scalar_key_result.hamming_weights.size(0) == 3
            && scalar_key_result.hamming_weights.size(1) == 2,
            "leakage HW shape mismatch")) {
        return 1;
    }
    for (std::int64_t row = 0; row < 3; ++row) {
        if (!expect_tensor_row(
                scalar_key_result.values,
                scalar_key_result.hamming_weights,
                row,
                Byte{0x2b},
                plaintext_bytes.data_ptr<Byte>()[row])) {
            return 1;
        }
    }

    auto torch_scalar_key = torch::tensor(0x2b, options);
    auto torch_scalar_result = leakflow::crypto::aes::first_round_sbox_leakage(torch_scalar_key, plaintext_bytes);
    if (!expect(torch::equal(torch_scalar_result.values, scalar_key_result.values),
            "Torch scalar key result values mismatch")) {
        return 1;
    }
    if (!expect(torch::equal(torch_scalar_result.hamming_weights, scalar_key_result.hamming_weights),
            "Torch scalar key result HW mismatch")) {
        return 1;
    }

    auto key_bytes = torch::tensor({0x2b, 0x00, 0xff}, options);
    auto paired_result = leakflow::crypto::aes::first_round_sbox_leakage(key_bytes, plaintext_bytes);
    for (std::int64_t row = 0; row < 3; ++row) {
        if (!expect_tensor_row(
                paired_result.values,
                paired_result.hamming_weights,
                row,
                key_bytes.data_ptr<Byte>()[row],
                plaintext_bytes.data_ptr<Byte>()[row])) {
            return 1;
        }
    }

    auto plaintext_blocks = torch::tensor(
        {{0x00, 0x32, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
             0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f},
            {0x10, 0xff, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f}},
        options);
    auto key_blocks = torch::tensor(
        {{0x00, 0x2b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
            {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
        options);

    auto block_result = leakflow::crypto::aes::first_round_sbox_leakage_at(key_blocks, plaintext_blocks, 1);
    if (!expect_tensor_row(block_result.values, block_result.hamming_weights, 0, Byte{0x2b}, Byte{0x32})) {
        return 1;
    }
    if (!expect_tensor_row(block_result.values, block_result.hamming_weights, 1, Byte{0x00}, Byte{0xff})) {
        return 1;
    }

    auto scalar_block_result = leakflow::crypto::aes::first_round_sbox_leakage_at(Byte{0x2b}, plaintext_blocks, 1);
    if (!expect_tensor_row(scalar_block_result.values, scalar_block_result.hamming_weights, 0, Byte{0x2b}, Byte{0x32})) {
        return 1;
    }
    if (!expect_tensor_row(scalar_block_result.values, scalar_block_result.hamming_weights, 1, Byte{0x2b}, Byte{0xff})) {
        return 1;
    }

    const std::array<std::size_t, 2> byte_indexes{1, 0};
    auto multi_block_result = leakflow::crypto::aes::first_round_sbox_leakage_at(
        key_blocks,
        plaintext_blocks,
        byte_indexes);
    if (!expect(multi_block_result.values.dim() == 3 && multi_block_result.values.size(0) == 2
            && multi_block_result.values.size(1) == 2 && multi_block_result.values.size(2) == 2,
            "multi-index leakage values shape mismatch")) {
        return 1;
    }
    if (!expect(multi_block_result.hamming_weights.dim() == 3 && multi_block_result.hamming_weights.size(0) == 2
            && multi_block_result.hamming_weights.size(1) == 2 && multi_block_result.hamming_weights.size(2) == 2,
            "multi-index leakage HW shape mismatch")) {
        return 1;
    }
    if (!expect_tensor_plane_row(multi_block_result.values, multi_block_result.hamming_weights, 0, 0, Byte{0x2b}, Byte{0x32})) {
        return 1;
    }
    if (!expect_tensor_plane_row(multi_block_result.values, multi_block_result.hamming_weights, 0, 1, Byte{0x00}, Byte{0xff})) {
        return 1;
    }
    if (!expect_tensor_plane_row(multi_block_result.values, multi_block_result.hamming_weights, 1, 0, Byte{0x00}, Byte{0x00})) {
        return 1;
    }
    if (!expect_tensor_plane_row(multi_block_result.values, multi_block_result.hamming_weights, 1, 1, Byte{0x00}, Byte{0x10})) {
        return 1;
    }

    auto scalar_multi_block_result = leakflow::crypto::aes::first_round_sbox_leakage_at(
        Byte{0x2b},
        plaintext_blocks,
        byte_indexes);
    if (!expect_tensor_plane_row(
            scalar_multi_block_result.values,
            scalar_multi_block_result.hamming_weights,
            0,
            0,
            Byte{0x2b},
            Byte{0x32})) {
        return 1;
    }
    if (!expect_tensor_plane_row(
            scalar_multi_block_result.values,
            scalar_multi_block_result.hamming_weights,
            1,
            1,
            Byte{0x2b},
            Byte{0x10})) {
        return 1;
    }

    const std::vector<std::string> channel_names{
        leakflow::crypto::aes::first_round_leakage_channel_hw_m,
        leakflow::crypto::aes::first_round_leakage_channel_hw_m_xor_k,
        leakflow::crypto::aes::first_round_leakage_channel_hw_y,
    };
    const auto parsed_channels = leakflow::crypto::aes::parse_first_round_leakage_channels(channel_names);
    const auto parsed_channel_span =
        std::span<const leakflow::crypto::aes::FirstRoundLeakageChannel>(
            parsed_channels.data(),
            parsed_channels.size());
    if (!expect(parsed_channels.size() == 3, "AES leakage channel parsing changed")) {
        return 1;
    }
    if (!expect(
            leakflow::crypto::aes::first_round_leakage_channels_metadata(parsed_channel_span)
                == "HW(m),HW(m_xor_k),HW(y)",
            "AES leakage channel metadata changed")) {
        return 1;
    }
    if (!expect(
            leakflow::crypto::aes::first_round_leakage_channel_dependencies_metadata(parsed_channel_span)
                == "false,true,true",
            "AES leakage channel dependency metadata changed")) {
        return 1;
    }
    if (!expect(
            leakflow::crypto::aes::first_round_leakage_channels_depend_on_key(parsed_channel_span),
            "AES leakage channel dependency detection changed")) {
        return 1;
    }

    auto shared_known_leakage = leakflow::crypto::aes::first_round_leakage_at(
        key_blocks,
        plaintext_blocks,
        byte_indexes,
        parsed_channel_span);
    if (!expect(shared_known_leakage.dim() == 3 && shared_known_leakage.size(0) == 2
            && shared_known_leakage.size(1) == 2 && shared_known_leakage.size(2) == 3,
            "shared known-key leakage shape mismatch")) {
        return 1;
    }
    const auto known_0 = leakflow::crypto::aes::first_round_sbox_leakage(Byte{0x2b}, Byte{0x32});
    if (!expect_leakage_channel(
            shared_known_leakage,
            0,
            0,
            0,
            leakflow::crypto::hamming_weight(Byte{0x32}),
            "shared known-key HW(m) mismatch")) {
        return 1;
    }
    if (!expect_leakage_channel(
            shared_known_leakage,
            0,
            0,
            1,
            leakflow::crypto::hamming_weight(Byte{0x32 ^ 0x2b}),
            "shared known-key HW(m_xor_k) mismatch")) {
        return 1;
    }
    if (!expect_leakage_channel(
            shared_known_leakage,
            0,
            0,
            2,
            known_0.hw_y,
            "shared known-key HW(y) mismatch")) {
        return 1;
    }

    const std::vector<leakflow::crypto::aes::FirstRoundLeakageChannel> hw_m_channels{
        leakflow::crypto::aes::FirstRoundLeakageChannel::HwM,
    };
    const auto hw_m_channel_span =
        std::span<const leakflow::crypto::aes::FirstRoundLeakageChannel>(
            hw_m_channels.data(),
            hw_m_channels.size());
    auto plaintext_only_leakage = leakflow::crypto::aes::first_round_leakage_at(
        plaintext_blocks,
        byte_indexes,
        hw_m_channel_span);
    if (!expect(plaintext_only_leakage.dim() == 3 && plaintext_only_leakage.size(0) == 2
            && plaintext_only_leakage.size(1) == 2 && plaintext_only_leakage.size(2) == 1,
            "shared plaintext-only leakage shape mismatch")) {
        return 1;
    }
    if (!expect_leakage_channel(
            plaintext_only_leakage,
            1,
            1,
            0,
            leakflow::crypto::hamming_weight(Byte{0x10}),
            "shared plaintext-only HW(m) mismatch")) {
        return 1;
    }

    auto guess_values = torch::tensor({0x00, 0x2b}, options);
    auto hypotheses = leakflow::crypto::aes::first_round_leakage_hypotheses_at(
        guess_values,
        plaintext_blocks,
        byte_indexes,
        parsed_channel_span);
    if (!expect(hypotheses.dim() == 4 && hypotheses.size(0) == 2
            && hypotheses.size(1) == 2 && hypotheses.size(2) == 2
            && hypotheses.size(3) == 3,
            "shared guess-domain hypothesis shape mismatch")) {
        return 1;
    }
    const auto scalar_key = torch::tensor(0x2b, options);
    auto scalar_known_leakage = leakflow::crypto::aes::first_round_leakage_at(
        scalar_key,
        plaintext_blocks,
        byte_indexes,
        parsed_channel_span);
    if (!expect(
            torch::equal(hypotheses.select(1, 1).contiguous(), scalar_known_leakage),
            "guess-domain hypothesis did not agree with known-key leakage for contained guess")) {
        return 1;
    }
    if (!expect_hypothesis_channel(
            hypotheses,
            0,
            0,
            0,
            1,
            leakflow::crypto::hamming_weight(Byte{0x32}),
            "guess-domain HW(m_xor_k) mismatch for zero guess")) {
        return 1;
    }
    if (!expect_hypothesis_channel(
            hypotheses,
            0,
            1,
            0,
            2,
            known_0.hw_y,
            "guess-domain HW(y) mismatch for contained guess")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([] {
            const std::vector<std::string> duplicate_channels{
                leakflow::crypto::aes::first_round_leakage_channel_hw_m,
                leakflow::crypto::aes::first_round_leakage_channel_hw_m,
            };
            (void)leakflow::crypto::aes::parse_first_round_leakage_channels(duplicate_channels);
        }),
            "AES leakage channel parsing should reject duplicates")) {
        return 1;
    }
    if (!expect(throws_exception<std::invalid_argument>([&plaintext_blocks, &byte_indexes, &parsed_channel_span] {
            (void)leakflow::crypto::aes::first_round_leakage_at(
                plaintext_blocks,
                byte_indexes,
                parsed_channel_span);
        }),
            "shared known-key leakage should reject key-dependent channels without keys")) {
        return 1;
    }
    if (!expect(throws_exception<std::invalid_argument>([&plaintext_blocks, &byte_indexes, &parsed_channel_span, &options] {
            const auto empty_guesses = torch::empty({0}, options);
            (void)leakflow::crypto::aes::first_round_leakage_hypotheses_at(
                empty_guesses,
                plaintext_blocks,
                byte_indexes,
                parsed_channel_span);
        }),
            "guess-domain leakage should reject empty guess domains")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([&plaintext_bytes] {
            (void)leakflow::crypto::aes::first_round_sbox_leakage(
                torch::tensor({0x2b, 0x00}, torch::TensorOptions().dtype(torch::kUInt8)),
                plaintext_bytes);
        }),
            "AES leakage should reject key vectors with the wrong length")) {
        return 1;
    }
    if (!expect(throws_exception<std::invalid_argument>([&plaintext_blocks] {
            (void)leakflow::crypto::aes::first_round_sbox_leakage_at(Byte{0x2b}, plaintext_blocks, 16);
        }),
            "AES leakage should reject byte_index outside [0,15]")) {
        return 1;
    }
    if (!expect(throws_exception<std::invalid_argument>([&plaintext_blocks] {
            const std::array<std::size_t, 2> bad_indexes{0, 16};
            (void)leakflow::crypto::aes::first_round_sbox_leakage_at(Byte{0x2b}, plaintext_blocks, bad_indexes);
        }),
            "AES leakage should reject byte_indexes outside [0,15]")) {
        return 1;
    }

    return 0;
}
