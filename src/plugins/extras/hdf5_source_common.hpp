#pragma once

#include "leakflow/core/element.hpp"
#include "leakflow/extras/tensor_dataset_reader.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace leakflow::plugins::extras::detail {

inline constexpr std::string_view traces_path = "/traces";
inline constexpr std::string_view plaintexts_path = "/plaintexts";
inline constexpr std::string_view keys_path = "/keys";
inline constexpr std::string_view ciphertexts_path = "/ciphertexts";
inline constexpr std::string_view countermeasures_prefix = "/countermeasures/";

struct Hdf5SourceOptions {
  std::filesystem::path path;
  std::uint64_t row_start = 0;
  std::uint64_t row_count = 0;
  std::uint64_t io_batch_rows = 256;
  std::string device = "cpu";
};

struct AggregateReadProgress {
  std::string array_path;
  std::uint64_t logical_bytes_read = 0;
  std::uint64_t total_logical_bytes = 0;
  std::uint64_t rows_read = 0;
  std::uint64_t total_rows = 0;
};

using AggregateReadProgressCallback =
    std::function<void(const AggregateReadProgress &)>;

[[nodiscard]] Hdf5SourceOptions source_options(const Element &element);
[[nodiscard]] std::uint64_t
aligned_row_count(const leakflow::extras::TensorDatasetDescriptor &descriptor);
[[nodiscard]] std::uint64_t
selected_row_count(const leakflow::extras::TensorDatasetDescriptor &descriptor,
                   std::uint64_t start, std::uint64_t count);

[[nodiscard]] ElementOutputs
read_hdf5_outputs(const leakflow::extras::TensorDatasetReader &reader,
                  const Hdf5SourceOptions &options,
                  AggregateReadProgressCallback progress = {});

} // namespace leakflow::plugins::extras::detail
