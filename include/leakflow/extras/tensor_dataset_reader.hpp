#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow::extras {

enum class TensorDatasetDType {
  UInt8,
  Float32,
};

[[nodiscard]] std::string_view
tensor_dataset_dtype_name(TensorDatasetDType dtype) noexcept;

using TensorDatasetAttributes = std::map<std::string, std::string>;

struct TensorRowSelection {
  std::uint64_t start = 0;
  std::optional<std::uint64_t> count = std::nullopt;
};

struct TensorArrayDescriptor {
  std::string path;
  TensorDatasetDType dtype = TensorDatasetDType::UInt8;
  std::vector<std::int64_t> shape;
  bool row_aligned = false;
  TensorDatasetAttributes attributes;

  [[nodiscard]] std::uint64_t element_size_bytes() const noexcept;
  [[nodiscard]] std::uint64_t logical_bytes() const;
  [[nodiscard]] std::uint64_t
  selected_logical_bytes(const TensorRowSelection &rows) const;
  [[nodiscard]] std::vector<std::int64_t>
  selected_shape(const TensorRowSelection &rows) const;
};

struct TensorGroupDescriptor {
  std::string path;
  TensorDatasetAttributes attributes;
};

struct TensorDatasetDescriptor {
  std::string storage_format;
  std::filesystem::path path;
  std::vector<TensorGroupDescriptor> groups;
  std::vector<TensorArrayDescriptor> arrays;

  [[nodiscard]] const TensorGroupDescriptor *
  find_group(std::string_view path) const noexcept;
  [[nodiscard]] const TensorArrayDescriptor *
  find_array(std::string_view path) const noexcept;
};

struct TensorReadOptions {
  TensorRowSelection rows{};
  std::uint64_t io_batch_rows = 256;
};

struct TensorReadProgress {
  std::string array_path;
  std::uint64_t logical_bytes_read = 0;
  std::uint64_t total_logical_bytes = 0;
  std::uint64_t rows_read = 0;
  std::uint64_t total_rows = 0;

  [[nodiscard]] double fraction() const noexcept;
};

using TensorReadProgressCallback =
    std::function<void(const TensorReadProgress &)>;

class TensorDatasetReader {
public:
  virtual ~TensorDatasetReader() = default;

  [[nodiscard]] virtual const TensorDatasetDescriptor &
  descriptor() const noexcept = 0;

  [[nodiscard]] virtual torch::Tensor
  read_tensor(std::string_view path, const TensorReadOptions &options = {},
              TensorReadProgressCallback progress = {}) const = 0;
};

} // namespace leakflow::extras
