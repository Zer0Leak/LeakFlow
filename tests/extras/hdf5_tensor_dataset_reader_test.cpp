#include "leakflow/extras/hdf5_tensor_dataset_reader.hpp"

#include <hdf5.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

template <typename Exception, typename Function>
bool throws_exception(Function &&function) {
  try {
    std::forward<Function>(function)();
  } catch (const Exception &) {
    return true;
  }
  return false;
}

void check(herr_t status, const char *operation) {
  if (status < 0) {
    throw std::runtime_error(std::string("test HDF5 operation failed: ") +
                             operation);
  }
}

hid_t check_id(hid_t id, const char *operation) {
  if (id < 0) {
    throw std::runtime_error(std::string("test HDF5 operation failed: ") +
                             operation);
  }
  return id;
}

void write_string_attribute(hid_t object, const char *name,
                            const std::string &value) {
  const auto space =
      check_id(H5Screate(H5S_SCALAR), "create scalar attribute space");
  const auto type = check_id(H5Tcopy(H5T_C_S1), "copy string type");
  check(H5Tset_size(type, value.size() + 1), "set string width");
  check(H5Tset_strpad(type, H5T_STR_NULLTERM), "set string padding");
  const auto attribute =
      check_id(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
               "create string attribute");
  check(H5Awrite(attribute, type, value.c_str()), "write string attribute");
  check(H5Aclose(attribute), "close string attribute");
  check(H5Tclose(type), "close string type");
  check(H5Sclose(space), "close string attribute space");
}

void write_integer_attribute(hid_t object, const char *name,
                             std::int64_t value) {
  const auto space =
      check_id(H5Screate(H5S_SCALAR), "create integer attribute space");
  const auto attribute = check_id(
      H5Acreate2(object, name, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT),
      "create integer attribute");
  check(H5Awrite(attribute, H5T_NATIVE_INT64, &value),
        "write integer attribute");
  check(H5Aclose(attribute), "close integer attribute");
  check(H5Sclose(space), "close integer attribute space");
}

void write_float_attribute(hid_t object, const char *name, double value) {
  const auto space =
      check_id(H5Screate(H5S_SCALAR), "create float attribute space");
  const auto attribute = check_id(
      H5Acreate2(object, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT),
      "create float attribute");
  check(H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value),
        "write float attribute");
  check(H5Aclose(attribute), "close float attribute");
  check(H5Sclose(space), "close float attribute space");
}

void write_bool_attribute(hid_t object, const char *name, bool value) {
  auto false_value = std::uint8_t{0};
  auto true_value = std::uint8_t{1};
  const auto file_type =
      check_id(H5Tenum_create(H5T_STD_U8LE), "create file bool enum");
  check(H5Tenum_insert(file_type, "FALSE", &false_value),
        "insert false file enum value");
  check(H5Tenum_insert(file_type, "TRUE", &true_value),
        "insert true file enum value");
  const auto memory_type =
      check_id(H5Tenum_create(H5T_NATIVE_UINT8), "create memory bool enum");
  check(H5Tenum_insert(memory_type, "FALSE", &false_value),
        "insert false memory enum value");
  check(H5Tenum_insert(memory_type, "TRUE", &true_value),
        "insert true memory enum value");
  const auto space =
      check_id(H5Screate(H5S_SCALAR), "create bool attribute space");
  const auto attribute = check_id(
      H5Acreate2(object, name, file_type, space, H5P_DEFAULT, H5P_DEFAULT),
      "create bool attribute");
  const auto encoded = static_cast<std::uint8_t>(value);
  check(H5Awrite(attribute, memory_type, &encoded), "write bool attribute");
  check(H5Aclose(attribute), "close bool attribute");
  check(H5Sclose(space), "close bool attribute space");
  check(H5Tclose(memory_type), "close memory bool enum");
  check(H5Tclose(file_type), "close file bool enum");
}

void create_group(hid_t file, const char *path) {
  const auto group =
      check_id(H5Gcreate2(file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
               "create group");
  check(H5Gclose(group), "close group");
}

template <typename Value>
void write_dataset(hid_t file, const char *path, hid_t file_type,
                   hid_t memory_type, const std::vector<hsize_t> &shape,
                   const std::vector<Value> &values, bool row_aligned,
                   const char *axes) {
  const auto space = check_id(
      H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr),
      "create dataset space");
  const auto dataset =
      check_id(H5Dcreate2(file, path, file_type, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT),
               "create dataset");
  check(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 values.data()),
        "write dataset");
  write_bool_attribute(dataset, "leakflow.row_aligned", row_aligned);
  write_string_attribute(dataset, "tensor.axes", axes);
  check(H5Dclose(dataset), "close dataset");
  check(H5Sclose(space), "close dataset space");
}

std::filesystem::path unique_temp_path(const char *suffix) {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         (std::string("leakflow_hdf5_reader_") + std::to_string(stamp) +
          suffix);
}

struct TempFiles {
  std::vector<std::filesystem::path> paths;
  ~TempFiles() {
    for (const auto &path : paths) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  }
};

void make_supported_fixture(const std::filesystem::path &path) {
  const auto file =
      check_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
               "create fixture");
  write_string_attribute(file, "leakflow.schema",
                         "leakflow.sca.tensor-dataset");
  write_string_attribute(file, "leakflow.schema.version", "1");

  create_group(file, "/metadata");
  const auto metadata =
      check_id(H5Gopen2(file, "/metadata", H5P_DEFAULT), "open metadata group");
  write_string_attribute(metadata, "capture.source", "ChipWhisperer");
  write_integer_attribute(metadata, "capture.scope.gain.setting", 22);
  write_float_attribute(metadata, "capture.sample_rate_hz", 29454545.454545453);
  check(H5Gclose(metadata), "close metadata group");

  create_group(file, "/countermeasures");
  create_group(file, "/countermeasures/jitter");
  const auto jitter =
      check_id(H5Gopen2(file, "/countermeasures/jitter", H5P_DEFAULT),
               "open jitter group");
  write_string_attribute(jitter, "type", "global-initial");
  check(H5Gclose(jitter), "close jitter group");
  create_group(file, "/countermeasures/jitter/parameters");

  const std::vector<float> traces{
      0.0F,  1.0F,  2.0F,  10.0F, 11.0F, 12.0F, 20.0F, 21.0F,
      22.0F, 30.0F, 31.0F, 32.0F, 40.0F, 41.0F, 42.0F,
  };
  const std::vector<std::uint8_t> plaintexts{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<std::uint8_t> keys{0x2b, 0x7e};
  const std::vector<std::uint8_t> ciphertexts{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  const std::vector<std::uint8_t> loop_iterations{0, 1, 2, 3, 4};

  write_dataset(file, "/traces", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {5, 3},
                traces, true, "trace,sample");
  write_dataset(file, "/plaintexts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {5, 2},
                plaintexts, true, "trace,byte");
  write_dataset(file, "/keys", H5T_STD_U8LE, H5T_NATIVE_UINT8, {2}, keys, false,
                "key_byte");
  write_dataset(file, "/ciphertexts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {5, 2},
                ciphertexts, true, "trace,byte");
  write_dataset(file, "/countermeasures/jitter/parameters/loop_iterations",
                H5T_STD_U8LE, H5T_NATIVE_UINT8, {5}, loop_iterations, true,
                "trace");
  check(H5Fclose(file), "close fixture");
}

void make_unsupported_fixture(const std::filesystem::path &path) {
  const auto file =
      check_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
               "create unsupported fixture");
  const std::vector<std::int16_t> values{1, 2};
  write_dataset(file, "/unsupported", H5T_STD_I16LE, H5T_NATIVE_INT16, {2},
                values, false, "value");
  check(H5Fclose(file), "close unsupported fixture");
}

void make_group_cycle_fixture(const std::filesystem::path &path) {
  const auto file =
      check_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
               "create group-cycle fixture");
  create_group(file, "/cycle");
  check(H5Lcreate_hard(file, "/cycle", file, "/cycle/self", H5P_DEFAULT,
                       H5P_DEFAULT),
        "create hard-linked group cycle");
  check(H5Fclose(file), "close group-cycle fixture");
}

void make_soft_link_fixture(const std::filesystem::path &path) {
  const auto file =
      check_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
               "create soft-link fixture");
  create_group(file, "/target");
  check(
      H5Lcreate_soft("/target", file, "/soft_target", H5P_DEFAULT, H5P_DEFAULT),
      "create soft link");
  check(H5Fclose(file), "close soft-link fixture");
}

void make_external_link_fixture(const std::filesystem::path &path) {
  const auto file =
      check_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
               "create external-link fixture");
  check(H5Lcreate_external("external.h5", "/target", file, "/external_target",
                           H5P_DEFAULT, H5P_DEFAULT),
        "create external link");
  check(H5Fclose(file), "close external-link fixture");
}

} // namespace

int main() {
  TempFiles cleanup;
  const auto fixture_path = unique_temp_path(".h5");
  const auto unsupported_path = unique_temp_path("_unsupported.h5");
  const auto group_cycle_path = unique_temp_path("_group_cycle.h5");
  const auto soft_link_path = unique_temp_path("_soft_link.h5");
  const auto external_link_path = unique_temp_path("_external_link.h5");
  cleanup.paths = {fixture_path, unsupported_path, group_cycle_path,
                   soft_link_path, external_link_path};
  make_supported_fixture(fixture_path);
  make_unsupported_fixture(unsupported_path);
  make_group_cycle_fixture(group_cycle_path);
  make_soft_link_fixture(soft_link_path);
  make_external_link_fixture(external_link_path);

  leakflow::extras::Hdf5TensorDatasetReader reader(fixture_path);
  const auto &descriptor = reader.descriptor();
  if (!expect(descriptor.storage_format == "hdf5",
              "HDF5 storage format mismatch")) {
    return 1;
  }
  if (!expect(descriptor.path == fixture_path,
              "HDF5 descriptor path mismatch")) {
    return 1;
  }
  if (!expect(descriptor.find_group("/") != nullptr,
              "root group was not inspected")) {
    return 1;
  }
  if (!expect(descriptor.find_group("/")->attributes.at("leakflow.schema") ==
                  "leakflow.sca.tensor-dataset",
              "root schema metadata mismatch")) {
    return 1;
  }
  const auto *metadata = descriptor.find_group("/metadata");
  if (!expect(metadata != nullptr, "metadata group was not inspected")) {
    return 1;
  }
  if (!expect(metadata->attributes.at("capture.source") == "ChipWhisperer",
              "string metadata mismatch")) {
    return 1;
  }
  if (!expect(metadata->attributes.at("capture.scope.gain.setting") == "22",
              "integer metadata mismatch")) {
    return 1;
  }
  if (!expect(metadata->attributes.contains("capture.sample_rate_hz"),
              "floating metadata was not inspected")) {
    return 1;
  }
  const auto *jitter = descriptor.find_group("/countermeasures/jitter");
  if (!expect(jitter != nullptr &&
                  jitter->attributes.at("type") == "global-initial",
              "nested group metadata mismatch")) {
    return 1;
  }
  if (!expect(descriptor.arrays.size() == 5,
              "recursive HDF5 array listing mismatch")) {
    return 1;
  }

  const auto *traces = descriptor.find_array("/traces");
  if (!expect(traces != nullptr, "traces array was not inspected")) {
    return 1;
  }
  if (!expect(traces->dtype == leakflow::extras::TensorDatasetDType::Float32,
              "traces dtype mismatch")) {
    return 1;
  }
  if (!expect(traces->shape == std::vector<std::int64_t>{5, 3},
              "traces shape mismatch")) {
    return 1;
  }
  if (!expect(traces->row_aligned, "traces row-alignment metadata mismatch")) {
    return 1;
  }
  if (!expect(traces->attributes.at("tensor.axes") == "trace,sample",
              "traces axes mismatch")) {
    return 1;
  }

  leakflow::extras::TensorReadOptions trace_options;
  trace_options.rows = {.start = 1, .count = 3};
  trace_options.io_batch_rows = 2;
  std::vector<leakflow::extras::TensorReadProgress> trace_progress;
  const auto trace_tensor = reader.read_tensor(
      "/traces", trace_options, [&trace_progress](const auto &update) {
        trace_progress.push_back(update);
      });
  if (!expect(trace_tensor.device().is_cpu() &&
                  trace_tensor.scalar_type() == torch::kFloat32,
              "trace tensor type/device mismatch")) {
    return 1;
  }
  if (!expect(trace_tensor.sizes().vec() == std::vector<std::int64_t>{3, 3},
              "selected trace shape mismatch")) {
    return 1;
  }
  if (!expect(trace_tensor.data_ptr<float>()[0] == 10.0F &&
                  trace_tensor.data_ptr<float>()[8] == 32.0F,
              "selected trace values mismatch")) {
    return 1;
  }
  if (!expect(trace_progress.size() == 2,
              "trace read did not report per-hyperslab progress")) {
    return 1;
  }
  if (!expect(trace_progress.front().logical_bytes_read == 24 &&
                  trace_progress.back().logical_bytes_read == 36 &&
                  trace_progress.back().total_logical_bytes == 36 &&
                  trace_progress.back().fraction() == 1.0,
              "trace logical-byte progress mismatch")) {
    return 1;
  }

  leakflow::extras::TensorReadOptions label_options;
  label_options.rows = {.start = 2, .count = 2};
  label_options.io_batch_rows = 1;
  std::vector<leakflow::extras::TensorReadProgress> label_progress;
  const auto labels =
      reader.read_tensor("/countermeasures/jitter/parameters/loop_iterations",
                         label_options, [&label_progress](const auto &update) {
                           label_progress.push_back(update);
                         });
  if (!expect(labels.sizes().vec() == std::vector<std::int64_t>{2} &&
                  labels.data_ptr<std::uint8_t>()[0] == 2 &&
                  labels.data_ptr<std::uint8_t>()[1] == 3,
              "rank-one row-aligned selection mismatch")) {
    return 1;
  }
  if (!expect(label_progress.size() == 2 &&
                  label_progress.back().total_logical_bytes == 2,
              "rank-one row-aligned progress mismatch")) {
    return 1;
  }

  leakflow::extras::TensorReadOptions key_options;
  key_options.rows = {.start = 999, .count = 1};
  key_options.io_batch_rows = 1;
  const auto key = reader.read_tensor("/keys", key_options);
  if (!expect(key.sizes().vec() == std::vector<std::int64_t>{2} &&
                  key.data_ptr<std::uint8_t>()[0] == 0x2b &&
                  key.data_ptr<std::uint8_t>()[1] == 0x7e,
              "fixed rank-one key should ignore row selection")) {
    return 1;
  }

  leakflow::extras::TensorReadOptions empty_options;
  empty_options.rows = {.start = 5, .count = 0};
  std::vector<leakflow::extras::TensorReadProgress> empty_progress;
  const auto empty = reader.read_tensor("/traces", empty_options,
                                        [&empty_progress](const auto &update) {
                                          empty_progress.push_back(update);
                                        });
  if (!expect(empty.sizes().vec() == std::vector<std::int64_t>{0, 3} &&
                  empty_progress.size() == 1 &&
                  empty_progress.front().fraction() == 1.0,
              "empty row selection behavior mismatch")) {
    return 1;
  }

  if (!expect(throws_exception<std::out_of_range>(
                  [&reader] { (void)reader.read_tensor("/missing"); }),
              "missing HDF5 array should throw")) {
    return 1;
  }
  if (!expect(throws_exception<std::invalid_argument>(
                  [&reader] { (void)reader.read_tensor("traces"); }),
              "relative HDF5 array path should throw")) {
    return 1;
  }
  if (!expect(throws_exception<std::invalid_argument>([&reader] {
                leakflow::extras::TensorReadOptions options;
                options.io_batch_rows = 0;
                (void)reader.read_tensor("/traces", options);
              }),
              "zero HDF5 I/O batch size should throw")) {
    return 1;
  }
  if (!expect(throws_exception<std::out_of_range>([&reader] {
                leakflow::extras::TensorReadOptions options;
                options.rows = {.start = 4, .count = 2};
                (void)reader.read_tensor("/traces", options);
              }),
              "out-of-range HDF5 row selection should throw")) {
    return 1;
  }
  if (!expect(throws_exception<std::invalid_argument>([&unsupported_path] {
                leakflow::extras::Hdf5TensorDatasetReader unsupported(
                    unsupported_path);
              }),
              "unsupported HDF5 tensor dtype should throw")) {
    return 1;
  }
  if (!expect(throws_exception<std::invalid_argument>([&group_cycle_path] {
                leakflow::extras::Hdf5TensorDatasetReader cyclic(
                    group_cycle_path);
              }),
              "hard-linked HDF5 group cycle should throw")) {
    return 1;
  }
  if (!expect(throws_exception<std::invalid_argument>([&soft_link_path] {
                leakflow::extras::Hdf5TensorDatasetReader soft_linked(
                    soft_link_path);
              }),
              "soft HDF5 link should throw")) {
    return 1;
  }
  if (!expect(throws_exception<std::invalid_argument>([&external_link_path] {
                leakflow::extras::Hdf5TensorDatasetReader external_linked(
                    external_link_path);
              }),
              "external HDF5 link should throw")) {
    return 1;
  }

  return 0;
}
