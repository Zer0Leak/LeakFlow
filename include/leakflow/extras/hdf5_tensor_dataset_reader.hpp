#pragma once

#include "leakflow/extras/tensor_dataset_reader.hpp"

#include <filesystem>
#include <memory>
#include <string_view>

namespace leakflow::extras {

class Hdf5TensorDatasetReader final : public TensorDatasetReader {
public:
  explicit Hdf5TensorDatasetReader(std::filesystem::path path);
  ~Hdf5TensorDatasetReader() override;

  Hdf5TensorDatasetReader(const Hdf5TensorDatasetReader &) = delete;
  Hdf5TensorDatasetReader &operator=(const Hdf5TensorDatasetReader &) = delete;
  Hdf5TensorDatasetReader(Hdf5TensorDatasetReader &&) noexcept;
  Hdf5TensorDatasetReader &operator=(Hdf5TensorDatasetReader &&) noexcept;

  [[nodiscard]] const TensorDatasetDescriptor &
  descriptor() const noexcept override;

  [[nodiscard]] torch::Tensor
  read_tensor(std::string_view path, const TensorReadOptions &options = {},
              TensorReadProgressCallback progress = {}) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace leakflow::extras
