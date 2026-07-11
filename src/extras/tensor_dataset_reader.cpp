#include "leakflow/extras/tensor_dataset_reader.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace leakflow::extras {
namespace {

[[nodiscard]] std::uint64_t checked_multiply(std::uint64_t left,
                                             std::uint64_t right,
                                             std::string_view context) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(context) +
                              " exceeds the supported byte-count range");
  }
  return left * right;
}

[[nodiscard]] std::uint64_t checked_dimension(std::int64_t dimension) {
  if (dimension < 0) {
    throw std::invalid_argument(
        "tensor dataset shape dimensions must be non-negative");
  }
  return static_cast<std::uint64_t>(dimension);
}

[[nodiscard]] std::uint64_t
selected_row_count(const TensorArrayDescriptor &descriptor,
                   const TensorRowSelection &rows) {
  if (descriptor.shape.empty()) {
    throw std::invalid_argument(
        "tensor dataset arrays must have rank one or greater");
  }

  // Fixed arrays such as a per-file key ignore trace-row selection. The
  // explicit row_aligned flag also supports rank-one per-trace labels.
  if (!descriptor.row_aligned) {
    return 1;
  }

  const auto available = checked_dimension(descriptor.shape.front());
  if (rows.start > available) {
    throw std::out_of_range(
        "tensor dataset row selection starts past the array extent");
  }

  const auto count = rows.count.value_or(available - rows.start);
  if (count > available - rows.start) {
    throw std::out_of_range(
        "tensor dataset row selection exceeds the array extent");
  }
  return count;
}

} // namespace

std::string_view tensor_dataset_dtype_name(TensorDatasetDType dtype) noexcept {
  switch (dtype) {
  case TensorDatasetDType::UInt8:
    return "uint8";
  case TensorDatasetDType::Float32:
    return "float32";
  }
  return "unknown";
}

std::uint64_t TensorArrayDescriptor::element_size_bytes() const noexcept {
  switch (dtype) {
  case TensorDatasetDType::UInt8:
    return sizeof(std::uint8_t);
  case TensorDatasetDType::Float32:
    return sizeof(float);
  }
  return 0;
}

std::uint64_t TensorArrayDescriptor::logical_bytes() const {
  if (shape.empty()) {
    throw std::invalid_argument(
        "tensor dataset arrays must have rank one or greater");
  }

  auto elements = std::uint64_t{1};
  for (const auto dimension : shape) {
    elements = checked_multiply(elements, checked_dimension(dimension),
                                "tensor dataset element count");
  }
  return checked_multiply(elements, element_size_bytes(),
                          "tensor dataset logical byte count");
}

std::vector<std::int64_t>
TensorArrayDescriptor::selected_shape(const TensorRowSelection &rows) const {
  (void)selected_row_count(*this, rows);
  auto result = shape;
  if (row_aligned) {
    result.front() = static_cast<std::int64_t>(selected_row_count(*this, rows));
  }
  return result;
}

std::uint64_t TensorArrayDescriptor::selected_logical_bytes(
    const TensorRowSelection &rows) const {
  const auto selected = selected_shape(rows);
  auto elements = std::uint64_t{1};
  for (const auto dimension : selected) {
    elements = checked_multiply(elements, checked_dimension(dimension),
                                "selected tensor element count");
  }
  return checked_multiply(elements, element_size_bytes(),
                          "selected tensor logical byte count");
}

const TensorGroupDescriptor *TensorDatasetDescriptor::find_group(
    std::string_view requested_path) const noexcept {
  const auto match =
      std::ranges::find_if(groups, [requested_path](const auto &group) {
        return group.path == requested_path;
      });
  return match == groups.end() ? nullptr : &*match;
}

const TensorArrayDescriptor *TensorDatasetDescriptor::find_array(
    std::string_view requested_path) const noexcept {
  const auto match =
      std::ranges::find_if(arrays, [requested_path](const auto &array) {
        return array.path == requested_path;
      });
  return match == arrays.end() ? nullptr : &*match;
}

double TensorReadProgress::fraction() const noexcept {
  if (total_logical_bytes == 0) {
    return 1.0;
  }
  return std::min(1.0, static_cast<double>(logical_bytes_read) /
                           static_cast<double>(total_logical_bytes));
}

} // namespace leakflow::extras
