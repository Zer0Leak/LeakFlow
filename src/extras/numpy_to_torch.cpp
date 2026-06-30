#include "leakflow/extras/numpy_to_torch.hpp"

#include "leakflow/log/logger.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace leakflow::extras {
namespace {

[[nodiscard]] torch::ScalarType torch_dtype_for_numpy(char dtype_code, unsigned word_size)
{
    switch (dtype_code) {
    case 'b':
        if (word_size == 1) {
            return torch::kBool;
        }
        break;
    case 'i':
        switch (word_size) {
        case 1:
            return torch::kInt8;
        case 2:
            return torch::kInt16;
        case 4:
            return torch::kInt32;
        case 8:
            return torch::kInt64;
        default:
            break;
        }
        break;
    case 'u':
        switch (word_size) {
        case 1:
            return torch::kUInt8;
        case 2:
            return torch::kUInt16;
        case 4:
            return torch::kUInt32;
        case 8:
            return torch::kUInt64;
        default:
            break;
        }
        break;
    case 'f':
        switch (word_size) {
        case 4:
            return torch::kFloat32;
        case 8:
            return torch::kFloat64;
        default:
            break;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("convert_numpy_to_torch received an unsupported NumPy dtype");
}

[[nodiscard]] std::vector<std::int64_t> torch_shape_for_numpy(const NumpyPayload& payload)
{
    std::vector<std::int64_t> shape;
    shape.reserve(payload.shape().size());
    for (const auto dimension : payload.shape()) {
        if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("convert_numpy_to_torch received a shape dimension too large for Torch");
        }
        shape.push_back(static_cast<std::int64_t>(dimension));
    }

    return shape;
}

[[nodiscard]] void* mutable_raw_data_pointer(const NumpyPayload& payload)
{
    return const_cast<void*>(payload.array().data<void>());
}

} // namespace

leakflow::base::TorchTensorPayload convert_numpy_to_torch(
    const NumpyPayload& payload,
    const NumpyToTorchOptions& options)
{
    if (payload.dtype_code() == '\0') {
        throw std::invalid_argument("convert_numpy_to_torch requires NumPy dtype metadata from load_npy");
    }

    const auto source_dtype = torch_dtype_for_numpy(payload.dtype_code(), payload.word_size());
    const auto target_dtype = options.target_dtype.value_or(source_dtype);
    const auto source_options = torch::TensorOptions().dtype(source_dtype).device(torch::kCPU);
    const auto target_options = torch::TensorOptions().dtype(target_dtype).device(options.target_device);

    auto tensor = torch::from_blob(mutable_raw_data_pointer(payload), torch_shape_for_numpy(payload), source_options)
                      .clone()
                      .to(target_options);

    log::LogRecord record{
        .level = log::LogLevel::Debug,
        .component = "extras",
        .message = "converted NumPy payload to Torch tensor",
        .fields = {
            {"source_dtype", payload.dtype_name()},
            {"source_shape", summary_list(payload.shape())},
            {"target_device", options.target_device.str()},
        },
    };
    if (options.target_dtype) {
        record.fields.emplace("target_dtype", c10::toString(*options.target_dtype));
    } else {
        record.fields.emplace("target_dtype", "preserve");
    }
    log::write(std::move(record));

    return leakflow::base::TorchTensorPayload(std::move(tensor));
}

} // namespace leakflow::extras
