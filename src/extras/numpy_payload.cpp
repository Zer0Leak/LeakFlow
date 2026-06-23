#include "leakflow/extras/numpy_payload.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/log/logger.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace leakflow::extras {
namespace {

struct NpyHeaderMetadata {
    char dtype_code = '\0';
    unsigned word_size = 0;
};

[[nodiscard]] bool contains_word_size(const std::vector<unsigned>& allowed, unsigned value)
{
    return std::ranges::find(allowed, value) != allowed.end();
}

void validate_simple_array_metadata(const cnpypp::NpyArray& array)
{
    if (array.memory_order != cnpypp::MemoryOrder::C) {
        throw std::invalid_argument("NumpyPayload supports C-contiguous arrays only");
    }
    if (!array.labels.empty()) {
        throw std::invalid_argument("NumpyPayload does not support structured NumPy arrays");
    }
    if (array.word_sizes.size() != 1) {
        throw std::invalid_argument("NumpyPayload requires a single plain NumPy dtype");
    }
}

void validate_supported_dtype(char dtype, unsigned word_size)
{
    switch (dtype) {
    case 'b':
        if (word_size == 1) {
            return;
        }
        break;
    case 'i':
    case 'u':
        if (contains_word_size({1, 2, 4, 8}, word_size)) {
            return;
        }
        break;
    case 'f':
        if (contains_word_size({4, 8}, word_size)) {
            return;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("load_npy supports bool, integer, unsigned integer, and float dtypes only");
}

[[nodiscard]] std::string numpy_dtype_name(char dtype_code, unsigned word_size)
{
    switch (dtype_code) {
    case 'b':
        if (word_size == 1) {
            return "bool";
        }
        break;
    case 'i':
        return "int" + std::to_string(word_size * 8);
    case 'u':
        return "uint" + std::to_string(word_size * 8);
    case 'f':
        return "float" + std::to_string(word_size * 8);
    case '\0':
        return "unknown";
    default:
        break;
    }

    return "unknown";
}

NpyHeaderMetadata validate_npy_header(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("load_npy could not open input file");
    }

    std::vector<unsigned> word_sizes;
    std::vector<char> data_types;
    std::vector<std::string> labels;
    std::vector<std::uint64_t> shape;
    cnpypp::MemoryOrder memory_order = cnpypp::MemoryOrder::C;
    cnpypp::parse_npy_header(file, word_sizes, data_types, labels, shape, memory_order);

    if (memory_order != cnpypp::MemoryOrder::C) {
        throw std::invalid_argument("load_npy supports C-contiguous arrays only");
    }
    if (!labels.empty()) {
        throw std::invalid_argument("load_npy does not support structured NumPy arrays");
    }
    if (data_types.size() != 1 || word_sizes.size() != 1) {
        throw std::invalid_argument("load_npy requires a single plain NumPy dtype");
    }

    validate_supported_dtype(data_types.front(), word_sizes.front());
    return {
        .dtype_code = data_types.front(),
        .word_size = word_sizes.front(),
    };
}

[[nodiscard]] std::string memory_order_name(cnpypp::MemoryOrder memory_order)
{
    switch (memory_order) {
    case cnpypp::MemoryOrder::C:
        return "C";
    case cnpypp::MemoryOrder::Fortran:
        return "Fortran";
    }

    return "unknown";
}

} // namespace

NumpyPayload::NumpyPayload(cnpypp::NpyArray array)
    : NumpyPayload(std::move(array), '\0')
{
}

NumpyPayload::NumpyPayload(cnpypp::NpyArray array, char dtype_code)
    : array_(std::move(array))
    , dtype_code_(dtype_code)
{
    validate_simple_array_metadata(array_);
    if (dtype_code_ != '\0') {
        validate_supported_dtype(dtype_code_, word_size());
    }
}

std::string NumpyPayload::type_name() const
{
    return numpy_array_caps_type;
}

void NumpyPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("dtype", dtype_name(), SummaryValueRole::TypeName);
    section.add_field("rank", summary_size(rank()), SummaryValueRole::Number);
    section.add_field("shape", summary_list(shape()), SummaryValueRole::Number);

    if (summary_level >= 2) {
        section.add_field("elements", summary_size(element_count()), SummaryValueRole::Size);
        section.add_field("word_size", summary_size(word_size()), SummaryValueRole::Size);
        section.add_field("byte_count", summary_size(byte_count()), SummaryValueRole::Size);
        section.add_field("memory_order", memory_order_name(memory_order()), SummaryValueRole::Text);
        section.add_field("labels", summary_size(array_.labels.size()), SummaryValueRole::Number);
    }
}

const cnpypp::NpyArray& NumpyPayload::array() const
{
    return array_;
}

cnpypp::NpyArray& NumpyPayload::array()
{
    return array_;
}

const std::vector<std::uint64_t>& NumpyPayload::shape() const
{
    return array_.shape;
}

cnpypp::MemoryOrder NumpyPayload::memory_order() const
{
    return array_.memory_order;
}

std::uint64_t NumpyPayload::rank() const
{
    return array_.shape.size();
}

std::uint64_t NumpyPayload::element_count() const
{
    return array_.num_vals;
}

std::uint64_t NumpyPayload::byte_count() const
{
    return array_.num_bytes();
}

unsigned NumpyPayload::word_size() const
{
    return array_.word_sizes.front();
}

char NumpyPayload::dtype_code() const
{
    return dtype_code_;
}

std::string NumpyPayload::dtype_name() const
{
    return numpy_dtype_name(dtype_code_, word_size());
}

std::string NumpyPayload::device_name() const
{
    return leakflow::base::cpu_device_caps_value;
}

Caps NumpyPayload::caps() const
{
    const auto& array_shape = shape();
    return leakflow::base::numeric_array_caps(
        numpy_array_caps_type,
        dtype_code_ == '\0' ? std::string() : dtype_name(),
        device_name(),
        rank(),
        leakflow::base::shape_caps_value(array_shape.data(), array_shape.size()));
}

NumpyPayload load_npy(const std::filesystem::path& path)
{
    const auto metadata = validate_npy_header(path);
    auto payload = NumpyPayload(cnpypp::npy_load(path.string()), metadata.dtype_code);
    if (payload.word_size() != metadata.word_size) {
        throw std::runtime_error("load_npy loaded NumPy dtype metadata inconsistent with parsed header");
    }

    log::LogRecord record{
        .level = log::LogLevel::Debug,
        .component = "extras",
        .message = "loaded NumPy payload",
        .fields = {
            {"dtype", payload.dtype_name()},
            {"path", path.string()},
            {"rank", std::to_string(payload.rank())},
            {"shape", summary_list(payload.shape())},
        },
    };
    log::write(std::move(record));

    return payload;
}

} // namespace leakflow::extras
