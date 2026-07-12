#include "hdf5_source_common.hpp"

#include "leakflow/base/torch_tensor_bundle_payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::extras::detail {
namespace {

inline constexpr std::string_view schema_name = "leakflow.sca.tensor-dataset";
inline constexpr std::string_view schema_version = "1";

[[nodiscard]] bool is_countermeasure_array(std::string_view path) {
  return path.starts_with(countermeasures_prefix);
}

[[nodiscard]] bool is_direct_output_array(std::string_view path) {
  return path == traces_path || path == plaintexts_path || path == keys_path ||
         path == ciphertexts_path;
}

[[nodiscard]] const std::string &
required_array_attribute(const leakflow::extras::TensorArrayDescriptor &array,
                         std::string_view name) {
  const auto attribute = array.attributes.find(std::string(name));
  if (attribute == array.attributes.end()) {
    throw std::invalid_argument("HDF5 tensor array " + array.path +
                                " is missing required " + std::string(name) +
                                " metadata");
  }
  return attribute->second;
}

[[nodiscard]] std::vector<std::string_view>
split_axes(const std::string &axes) {
  std::vector<std::string_view> result;
  auto remaining = std::string_view(axes);
  while (true) {
    const auto comma = remaining.find(',');
    const auto axis = remaining.substr(0, comma);
    if (axis.empty()) {
      throw std::invalid_argument("HDF5 tensor.axes contains an empty axis");
    }
    result.push_back(axis);
    if (comma == std::string_view::npos) {
      return result;
    }
    remaining.remove_prefix(comma + 1);
  }
}

[[nodiscard]] std::string payload_layout_for_axes(const std::string &axes) {
  const auto split = split_axes(axes);
  std::ostringstream output;
  for (std::size_t index = 0; index < split.size(); ++index) {
    if (index != 0) {
      output << '/';
    }
    output << split[index];
  }
  return output.str();
}

void validate_array_contract(
    const leakflow::extras::TensorArrayDescriptor &array,
    leakflow::extras::TensorDatasetDType dtype, std::size_t rank,
    bool row_aligned, std::string_view axes, std::string_view role) {
  const auto expected_alignment = row_aligned ? "true" : "false";
  if (array.dtype != dtype || array.shape.size() != rank ||
      array.row_aligned != row_aligned ||
      required_array_attribute(array, "leakflow.row_aligned") !=
          expected_alignment ||
      required_array_attribute(array, "tensor.axes") != axes ||
      required_array_attribute(array, "origin.role") != role) {
    throw std::invalid_argument("HDF5 tensor array " + array.path +
                                " violates the LeakFlow schema v1 contract");
  }
}

void validate_countermeasure_array(
    const leakflow::extras::TensorArrayDescriptor &array) {
  if (!array.row_aligned ||
      required_array_attribute(array, "leakflow.row_aligned") != "true" ||
      required_array_attribute(array, "origin.role") != "countermeasures") {
    throw std::invalid_argument("HDF5 countermeasure array " + array.path +
                                " must be aligned to trace rows");
  }
  const auto axes = split_axes(required_array_attribute(array, "tensor.axes"));
  if (axes.size() != array.shape.size() || axes.front() != "trace") {
    throw std::invalid_argument("HDF5 countermeasure array " + array.path +
                                " must declare trace as its leading axis");
  }
}

void validate_schema(
    const leakflow::extras::TensorDatasetDescriptor &descriptor) {
  const auto *root = descriptor.find_group("/");
  if (root == nullptr) {
    throw std::invalid_argument("HDF5 tensor dataset has no root group");
  }
  const auto name = root->attributes.find("leakflow.schema");
  const auto version = root->attributes.find("leakflow.schema.version");
  if (name == root->attributes.end() || name->second != schema_name) {
    throw std::invalid_argument(
        "HDF5 file is not a LeakFlow SCA tensor dataset");
  }
  if (version == root->attributes.end() || version->second != schema_version) {
    throw std::invalid_argument(
        "HDF5 tensor dataset schema version is unsupported");
  }

  for (const auto &array : descriptor.arrays) {
    if (array.path == traces_path) {
      validate_array_contract(array,
                              leakflow::extras::TensorDatasetDType::Float32, 2,
                              true, "trace,sample", "traces");
      if (array.shape[1] <= 0) {
        throw std::invalid_argument("HDF5 traces must contain samples");
      }
    } else if (array.path == plaintexts_path) {
      validate_array_contract(array,
                              leakflow::extras::TensorDatasetDType::UInt8, 2,
                              true, "trace,byte", "plaintexts");
      if (array.shape[1] != 16) {
        throw std::invalid_argument(
            "HDF5 AES plaintexts must contain 16 bytes");
      }
    } else if (array.path == keys_path) {
      validate_array_contract(array,
                              leakflow::extras::TensorDatasetDType::UInt8, 1,
                              false, "key_byte", "keys");
      if (array.shape[0] != 16) {
        throw std::invalid_argument("HDF5 AES keys must contain 16 bytes");
      }
    } else if (array.path == ciphertexts_path) {
      validate_array_contract(array,
                              leakflow::extras::TensorDatasetDType::UInt8, 2,
                              true, "trace,byte", "ciphertexts");
      if (array.shape[1] != 16) {
        throw std::invalid_argument(
            "HDF5 AES ciphertexts must contain 16 bytes");
      }
    } else if (is_countermeasure_array(array.path)) {
      validate_countermeasure_array(array);
    }
  }
}

[[nodiscard]] std::string string_property_or(const Element &element,
                                             std::string_view name,
                                             std::string fallback) {
  if (const auto value = element.property_as<std::string>(name)) {
    return *value;
  }
  return fallback;
}

[[nodiscard]] std::int64_t integer_property_or(const Element &element,
                                               std::string_view name,
                                               std::int64_t fallback) {
  if (const auto value = element.property_as<std::int64_t>(name)) {
    return *value;
  }
  return fallback;
}

[[nodiscard]] std::string_view output_pad_for(std::string_view path) {
  if (path == traces_path) {
    return "traces";
  }
  if (path == plaintexts_path) {
    return "plaintexts";
  }
  if (path == keys_path) {
    return "keys";
  }
  if (path == ciphertexts_path) {
    return "ciphertexts";
  }
  throw std::invalid_argument("HDF5 array has no direct output pad");
}

[[nodiscard]] std::string default_axes_for(std::string_view path) {
  if (path == traces_path) {
    return "trace,sample";
  }
  if (path == plaintexts_path || path == ciphertexts_path) {
    return "trace,byte";
  }
  if (path == keys_path) {
    return "key_byte";
  }
  if (is_countermeasure_array(path)) {
    return "trace";
  }
  return {};
}

[[nodiscard]] std::string default_role_for(std::string_view path) {
  if (path == traces_path) {
    return "traces";
  }
  if (path == plaintexts_path) {
    return "plaintexts";
  }
  if (path == keys_path) {
    return "keys";
  }
  if (path == ciphertexts_path) {
    return "ciphertexts";
  }
  return "countermeasures";
}

[[nodiscard]] std::string countermeasure_tensor_name(std::string_view path) {
  auto relative = path.substr(countermeasures_prefix.size());
  std::string result(relative);
  std::replace(result.begin(), result.end(), '/', '.');
  return result;
}

void copy_attributes(
    Buffer &buffer,
    const leakflow::extras::TensorDatasetAttributes &attributes) {
  for (const auto &[key, value] : attributes) {
    // `leakflow.*` attributes describe the storage schema itself. Buffer
    // metadata carries experiment/data facts; origin.file.* below records
    // the concrete storage provenance.
    if (!key.starts_with("leakflow.")) {
      buffer.set_metadata(key, value);
    }
  }
}

void copy_common_attributes(
    Buffer &buffer,
    const leakflow::extras::TensorDatasetAttributes &attributes) {
  for (const auto &[key, value] : attributes) {
    // payload.* describes one concrete output value. Role-specific payload
    // facts belong on that array, never on the common /metadata group.
    if (!key.starts_with("leakflow.") && !key.starts_with("payload.")) {
      buffer.set_metadata(key, value);
    }
  }
}

void copy_payload_attributes(
    Buffer &buffer,
    const leakflow::extras::TensorDatasetAttributes &attributes) {
  for (const auto &[key, value] : attributes) {
    if (key.starts_with("payload.")) {
      buffer.set_metadata(key, value);
    }
  }
}

void stamp_common_metadata(
    Buffer &buffer, const leakflow::extras::TensorDatasetDescriptor &descriptor,
    const Hdf5SourceOptions &options, std::uint64_t selected_rows) {
  if (const auto *metadata = descriptor.find_group("/metadata")) {
    copy_common_attributes(buffer, metadata->attributes);
  }

  buffer.set_metadata("origin.file.format", descriptor.storage_format);
  buffer.set_metadata("origin.file.path", options.path.string());
  buffer.set_metadata("origin.file.size",
                      std::to_string(std::filesystem::file_size(options.path)));
  buffer.set_metadata("origin.row.begin", std::to_string(options.row_start));
  buffer.set_metadata("origin.row.count", std::to_string(selected_rows));
  buffer.set_metadata("origin.row.total",
                      std::to_string(aligned_row_count(descriptor)));
}

[[nodiscard]] torch::Tensor move_to_device(torch::Tensor tensor,
                                           const std::string &device) {
  if (device.empty() || device == "cpu" || device == "preserve") {
    return tensor;
  }
  return tensor.to(torch::Device(device));
}

[[nodiscard]] leakflow::extras::TensorRowSelection
row_selection(const Hdf5SourceOptions &options) {
  return {
      .start = options.row_start,
      .count = options.row_count == 0
                   ? std::nullopt
                   : std::optional<std::uint64_t>(options.row_count),
  };
}

[[nodiscard]] std::vector<const leakflow::extras::TensorArrayDescriptor *>
selected_arrays(const leakflow::extras::TensorDatasetDescriptor &descriptor) {
  std::vector<const leakflow::extras::TensorArrayDescriptor *> result;
  for (const auto &array : descriptor.arrays) {
    if (is_direct_output_array(array.path) ||
        is_countermeasure_array(array.path)) {
      result.push_back(&array);
    }
  }
  return result;
}

[[nodiscard]] std::string countermeasure_layout(
    const std::vector<std::pair<const leakflow::extras::TensorArrayDescriptor *,
                                std::string>> &arrays) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < arrays.size(); ++index) {
    if (index != 0) {
      stream << ';';
    }
    const auto *descriptor = arrays[index].first;
    const auto axes = descriptor->attributes.contains("tensor.axes")
                          ? descriptor->attributes.at("tensor.axes")
                          : default_axes_for(descriptor->path);
    stream << arrays[index].second << '=' << payload_layout_for_axes(axes);
  }
  return stream.str();
}

} // namespace

Hdf5SourceOptions source_options(const Element &element) {
  const auto path = string_property_or(element, "path", "");
  if (path.empty()) {
    throw std::invalid_argument(element.element_type() +
                                " path property must not be empty");
  }

  const auto row_start = integer_property_or(element, "row_start", 0);
  const auto row_count = integer_property_or(element, "row_count", 0);
  const auto io_batch_rows = integer_property_or(element, "io_batch_rows", 256);
  if (row_start < 0 || row_count < 0 || io_batch_rows <= 0) {
    throw std::invalid_argument(element.element_type() +
                                " row properties are invalid");
  }

  return {
      .path = path,
      .row_start = static_cast<std::uint64_t>(row_start),
      .row_count = static_cast<std::uint64_t>(row_count),
      .io_batch_rows = static_cast<std::uint64_t>(io_batch_rows),
      .device = string_property_or(element, "device", "cpu"),
  };
}

std::uint64_t
aligned_row_count(const leakflow::extras::TensorDatasetDescriptor &descriptor) {
  validate_schema(descriptor);
  std::optional<std::uint64_t> rows;
  for (const auto &array : descriptor.arrays) {
    if ((!is_direct_output_array(array.path) &&
         !is_countermeasure_array(array.path)) ||
        !array.row_aligned) {
      continue;
    }
    if (array.shape.empty() || array.shape.front() < 0) {
      throw std::invalid_argument(
          "row-aligned HDF5 tensor has no valid leading trace axis: " +
          array.path);
    }
    const auto current = static_cast<std::uint64_t>(array.shape.front());
    if (rows && *rows != current) {
      throw std::invalid_argument(
          "row-aligned HDF5 tensors have inconsistent leading dimensions");
    }
    rows = current;
  }
  return rows.value_or(1);
}

std::uint64_t
selected_row_count(const leakflow::extras::TensorDatasetDescriptor &descriptor,
                   std::uint64_t start, std::uint64_t count) {
  const auto rows = aligned_row_count(descriptor);
  if (start > rows) {
    throw std::out_of_range("HDF5 row_start exceeds the available row count");
  }
  const auto available = rows - start;
  if (count == 0) {
    return available;
  }
  if (count > available) {
    throw std::out_of_range("HDF5 row_count exceeds the available row range");
  }
  return count;
}

ElementOutputs
read_hdf5_outputs(const leakflow::extras::TensorDatasetReader &reader,
                  const Hdf5SourceOptions &options,
                  AggregateReadProgressCallback progress) {
  const auto &descriptor = reader.descriptor();
  const auto rows =
      selected_row_count(descriptor, options.row_start, options.row_count);
  const auto arrays = selected_arrays(descriptor);
  if (arrays.empty()) {
    throw std::invalid_argument(
        "HDF5 tensor dataset contains no supported LeakFlow arrays");
  }

  const auto selection = row_selection(options);
  std::uint64_t total_bytes = 0;
  for (const auto *array : arrays) {
    total_bytes += array->selected_logical_bytes(selection);
  }

  ElementOutputs outputs;
  auto countermeasure_bundle =
      std::make_shared<leakflow::base::TorchTensorBundlePayload>();
  std::vector<
      std::pair<const leakflow::extras::TensorArrayDescriptor *, std::string>>
      countermeasure_arrays;
  std::uint64_t completed_bytes = 0;

  for (const auto *array : arrays) {
    const auto bytes_before = completed_bytes;
    const leakflow::extras::TensorReadOptions read_options{
        .rows = selection,
        .io_batch_rows = options.io_batch_rows,
    };
    auto tensor = reader.read_tensor(
        array->path, read_options,
        [&](const leakflow::extras::TensorReadProgress &update) {
          // Forward the caller's continue/abort decision to the reader so a
          // cancel unwinds mid-hyperslab as TensorReadCancelled. No callback
          // means never cancel.
          if (!progress) {
            return true;
          }
          return progress(AggregateReadProgress{
              .array_path = update.array_path,
              .logical_bytes_read = bytes_before + update.logical_bytes_read,
              .total_logical_bytes = total_bytes,
              .rows_read = update.rows_read,
              .total_rows = update.total_rows,
          });
        });
    completed_bytes += array->selected_logical_bytes(selection);
    tensor = move_to_device(std::move(tensor), options.device);

    if (is_countermeasure_array(array->path)) {
      auto name = countermeasure_tensor_name(array->path);
      countermeasure_bundle->set(name, std::move(tensor));
      countermeasure_arrays.emplace_back(array, std::move(name));
      continue;
    }

    auto payload =
        std::make_shared<leakflow::base::TorchTensorPayload>(std::move(tensor));
    Buffer buffer(payload->caps());
    stamp_common_metadata(buffer, descriptor, options, rows);
    copy_attributes(buffer, array->attributes);
    buffer.set_metadata("origin.hdf5.dataset", array->path);
    if (!buffer.has_metadata("origin.role")) {
      buffer.set_metadata("origin.role", default_role_for(array->path));
    }
    if (!buffer.has_metadata("tensor.axes")) {
      buffer.set_metadata("tensor.axes", default_axes_for(array->path));
    }
    buffer.set_payload(std::move(payload));
    buffer.set_metadata("payload.layout",
                        payload_layout_for_axes(buffer.metadata("tensor.axes")));
    outputs.emplace(std::string(output_pad_for(array->path)),
                    std::move(buffer));
  }

  if (countermeasure_bundle->size() != 0) {
    Buffer buffer(Caps("leakflow/torch-tensor-bundle"));
    stamp_common_metadata(buffer, descriptor, options, rows);
    for (const auto &[array, name] : countermeasure_arrays) {
      (void)name;
      copy_payload_attributes(buffer, array->attributes);
    }
    buffer.set_metadata("origin.role", "countermeasures");
    buffer.set_metadata("origin.hdf5.dataset", "/countermeasures");
    std::ostringstream names;
    const auto bundle_names = countermeasure_bundle->names();
    for (std::size_t index = 0; index < bundle_names.size(); ++index) {
      if (index != 0) {
        names << ',';
      }
      names << bundle_names[index];
    }
    buffer.set_metadata("payload.countermeasure.tensors", names.str());
    buffer.set_payload(std::move(countermeasure_bundle));
    buffer.set_metadata("payload.layout",
                        countermeasure_layout(countermeasure_arrays));
    outputs.emplace("countermeasures", std::move(buffer));
  }

  return outputs;
}

} // namespace leakflow::plugins::extras::detail
