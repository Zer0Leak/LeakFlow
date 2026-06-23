#include "leakflow/base/torch_tensor_payload.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/log/logger.hpp"

#include <c10/core/ScalarType.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace leakflow::base {
namespace {

void validate_tensor(const torch::Tensor& tensor)
{
    if (!tensor.defined()) {
        throw std::invalid_argument("TorchTensorPayload requires a defined tensor");
    }
    if (tensor.layout() != torch::kStrided) {
        throw std::invalid_argument("TorchTensorPayload requires a dense strided tensor");
    }
}

[[nodiscard]] std::string shape_to_string(c10::IntArrayRef values)
{
    std::vector<std::int64_t> vector_values(values.begin(), values.end());
    return summary_list_from_int_array(vector_values.data(), vector_values.size());
}

} // namespace

std::string torch_dtype_name(torch::ScalarType dtype)
{
    switch (dtype) {
    case c10::ScalarType::Byte:
        return "uint8";
    case c10::ScalarType::Char:
        return "int8";
    case c10::ScalarType::Short:
        return "int16";
    case c10::ScalarType::Int:
        return "int32";
    case c10::ScalarType::Long:
        return "int64";
    case c10::ScalarType::Half:
        return "float16";
    case c10::ScalarType::Float:
        return "float32";
    case c10::ScalarType::Double:
        return "float64";
    case c10::ScalarType::ComplexHalf:
        return "complex32";
    case c10::ScalarType::ComplexFloat:
        return "complex64";
    case c10::ScalarType::ComplexDouble:
        return "complex128";
    case c10::ScalarType::Bool:
        return "bool";
    case c10::ScalarType::QInt8:
        return "qint8";
    case c10::ScalarType::QUInt8:
        return "quint8";
    case c10::ScalarType::QInt32:
        return "qint32";
    case c10::ScalarType::BFloat16:
        return "bfloat16";
    case c10::ScalarType::QUInt4x2:
        return "quint4x2";
    case c10::ScalarType::QUInt2x4:
        return "quint2x4";
    case c10::ScalarType::Bits1x8:
        return "bits1x8";
    case c10::ScalarType::Bits2x4:
        return "bits2x4";
    case c10::ScalarType::Bits4x2:
        return "bits4x2";
    case c10::ScalarType::Bits8:
        return "bits8";
    case c10::ScalarType::Bits16:
        return "bits16";
    case c10::ScalarType::Float8_e5m2:
        return "float8_e5m2";
    case c10::ScalarType::Float8_e4m3fn:
        return "float8_e4m3fn";
    case c10::ScalarType::Float8_e5m2fnuz:
        return "float8_e5m2fnuz";
    case c10::ScalarType::Float8_e4m3fnuz:
        return "float8_e4m3fnuz";
    case c10::ScalarType::UInt16:
        return "uint16";
    case c10::ScalarType::UInt32:
        return "uint32";
    case c10::ScalarType::UInt64:
        return "uint64";
    case c10::ScalarType::UInt1:
        return "uint1";
    case c10::ScalarType::UInt2:
        return "uint2";
    case c10::ScalarType::UInt3:
        return "uint3";
    case c10::ScalarType::UInt4:
        return "uint4";
    case c10::ScalarType::UInt5:
        return "uint5";
    case c10::ScalarType::UInt6:
        return "uint6";
    case c10::ScalarType::UInt7:
        return "uint7";
    case c10::ScalarType::Int1:
        return "int1";
    case c10::ScalarType::Int2:
        return "int2";
    case c10::ScalarType::Int3:
        return "int3";
    case c10::ScalarType::Int4:
        return "int4";
    case c10::ScalarType::Int5:
        return "int5";
    case c10::ScalarType::Int6:
        return "int6";
    case c10::ScalarType::Int7:
        return "int7";
    case c10::ScalarType::Float8_e8m0fnu:
        return "float8_e8m0fnu";
    default:
        return c10::toString(dtype);
    }
}

TorchTensorPayload::TorchTensorPayload(torch::Tensor tensor)
    : tensor_(std::move(tensor))
{
    validate_tensor(tensor_);

    log::LogRecord record{
        .level = log::LogLevel::Trace,
        .component = "base",
        .message = "created Torch tensor payload",
        .fields = {
            {"dtype", dtype_name()},
            {"device", device_name()},
            {"rank", std::to_string(rank())},
            {"shape", shape_to_string(shape())},
        },
    };
    log::write(std::move(record));
}

std::string TorchTensorPayload::type_name() const
{
    return torch_tensor_caps_type;
}

void TorchTensorPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("dtype", dtype_name(), SummaryValueRole::TypeName);
    section.add_field("device", device_name(), SummaryValueRole::Text);
    section.add_field("rank", summary_integer(rank()), SummaryValueRole::Number);
    section.add_field("shape", shape_to_string(shape()), SummaryValueRole::Number);
    section.add_field("elements", summary_integer(element_count()), SummaryValueRole::Size);

    if (summary_level >= 2) {
        section.add_field("strides", shape_to_string(strides()), SummaryValueRole::Number);
        section.add_field("contiguous", summary_bool(is_contiguous()), SummaryValueRole::Boolean);
    }
}

const torch::Tensor& TorchTensorPayload::tensor() const
{
    return tensor_;
}

torch::Tensor& TorchTensorPayload::tensor()
{
    return tensor_;
}

torch::Device TorchTensorPayload::device() const
{
    return tensor_.device();
}

torch::ScalarType TorchTensorPayload::dtype() const
{
    return tensor_.scalar_type();
}

std::string TorchTensorPayload::dtype_name() const
{
    return torch_dtype_name(dtype());
}

std::string TorchTensorPayload::device_name() const
{
    return device().str();
}

c10::IntArrayRef TorchTensorPayload::shape() const
{
    return tensor_.sizes();
}

c10::IntArrayRef TorchTensorPayload::strides() const
{
    return tensor_.strides();
}

std::int64_t TorchTensorPayload::rank() const
{
    return tensor_.dim();
}

std::int64_t TorchTensorPayload::element_count() const
{
    return tensor_.numel();
}

bool TorchTensorPayload::is_cpu() const
{
    return tensor_.is_cpu();
}

bool TorchTensorPayload::is_cuda() const
{
    return tensor_.is_cuda();
}

bool TorchTensorPayload::is_contiguous() const
{
    return tensor_.is_contiguous();
}

Caps TorchTensorPayload::caps() const
{
    const auto tensor_shape = shape();
    return numeric_array_caps(
        torch_tensor_caps_type,
        dtype_name(),
        device_name(),
        static_cast<std::uint64_t>(rank()),
        shape_caps_value(tensor_shape.data(), tensor_shape.size()));
}

} // namespace leakflow::base
