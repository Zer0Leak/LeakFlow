#include "leakflow/base/torch_tensor_bundle_payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/progress_sink.hpp"
#include "leakflow/plugins/extras/fake_live_hdf5_src.hpp"
#include "leakflow/plugins/extras/hdf5_file_src.hpp"

#include <hdf5.h>

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
    throw std::runtime_error(std::string("HDF5 test operation failed: ") +
                             operation);
  }
}

hid_t check_id(hid_t id, const char *operation) {
  if (id < 0) {
    throw std::runtime_error(std::string("HDF5 test operation failed: ") +
                             operation);
  }
  return id;
}

void write_string_attribute(hid_t object, const char *name,
                            const std::string &value) {
  const auto space = check_id(H5Screate(H5S_SCALAR), "create attribute space");
  const auto type = check_id(H5Tcopy(H5T_C_S1), "copy string type");
  check(H5Tset_size(type, value.size() + 1), "set string width");
  check(H5Tset_strpad(type, H5T_STR_NULLTERM), "set string padding");
  const auto attribute =
      check_id(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
               "create string attribute");
  check(H5Awrite(attribute, type, value.c_str()), "write string attribute");
  check(H5Aclose(attribute), "close string attribute");
  check(H5Tclose(type), "close string type");
  check(H5Sclose(space), "close attribute space");
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
                   const char *axes, const char *role) {
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
  write_string_attribute(dataset, "leakflow.row_aligned",
                         row_aligned ? "true" : "false");
  write_string_attribute(dataset, "tensor.axes", axes);
  write_string_attribute(dataset, "origin.role", role);
  check(H5Dclose(dataset), "close dataset");
  check(H5Sclose(space), "close dataset space");
}

std::filesystem::path temp_path(const char *label) {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         (std::string("leakflow_hdf5_src_") + label + '_' +
          std::to_string(stamp) + ".h5");
}

void make_fixture(const std::filesystem::path &path, bool with_countermeasure,
                  bool traces_row_aligned = true) {
  const auto file =
      check_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
               "create file");
  write_string_attribute(file, "leakflow.schema",
                         "leakflow.sca.tensor-dataset");
  write_string_attribute(file, "leakflow.schema.version", "1");
  create_group(file, "/metadata");
  const auto metadata =
      check_id(H5Gopen2(file, "/metadata", H5P_DEFAULT), "open metadata");
  write_string_attribute(metadata, "capture.source", "ChipWhisperer");
  write_string_attribute(metadata, "capture.dataset.name",
                         with_countermeasure ? "unit_jitter" : "unit_sync");
  check(H5Gclose(metadata), "close metadata");

  const std::vector<float> traces{0.F,  1.F,  2.F,  10.F, 11.F,
                                  12.F, 20.F, 21.F, 22.F, 30.F,
                                  31.F, 32.F, 40.F, 41.F, 42.F};
  std::vector<std::uint8_t> plaintexts(5 * 16);
  std::vector<std::uint8_t> ciphertexts(5 * 16);
  std::vector<std::uint8_t> keys(16);
  for (std::size_t index = 0; index < plaintexts.size(); ++index) {
    plaintexts[index] = static_cast<std::uint8_t>(index);
    ciphertexts[index] = static_cast<std::uint8_t>(255 - index);
  }
  for (std::size_t index = 0; index < keys.size(); ++index) {
    keys[index] = static_cast<std::uint8_t>(index);
  }
  write_dataset(file, "/traces", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {5, 3},
                traces, traces_row_aligned, "trace,sample", "traces");
  const auto traces_dataset =
      check_id(H5Dopen2(file, "/traces", H5P_DEFAULT), "open traces");
  write_string_attribute(traces_dataset, "payload.leakage.inverted", "false");
  check(H5Dclose(traces_dataset), "close traces");
  write_dataset(file, "/plaintexts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {5, 16},
                plaintexts, true, "trace,byte", "plaintexts");
  write_dataset(file, "/keys", H5T_STD_U8LE, H5T_NATIVE_UINT8, {16}, keys,
                false, "key_byte", "keys");
  write_dataset(file, "/ciphertexts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {5, 16},
                ciphertexts, true, "trace,byte", "ciphertexts");

  if (with_countermeasure) {
    create_group(file, "/countermeasures");
    create_group(file, "/countermeasures/jitter");
    const auto jitter = check_id(
        H5Gopen2(file, "/countermeasures/jitter", H5P_DEFAULT), "open jitter");
    write_string_attribute(jitter, "type", "global-initial");
    write_string_attribute(jitter, "label_provenance", "derived-exact");
    check(H5Gclose(jitter), "close jitter");
    create_group(file, "/countermeasures/jitter/parameters");
    const std::vector<std::uint8_t> iterations{0, 1, 2, 3, 4};
    write_dataset(file, "/countermeasures/jitter/parameters/loop_iterations",
                  H5T_STD_U8LE, H5T_NATIVE_UINT8, {5}, iterations, true,
                  "trace", "countermeasures");
    const auto parameter = check_id(
        H5Dopen2(file, "/countermeasures/jitter/parameters/loop_iterations",
                 H5P_DEFAULT),
        "open jitter parameter");
    write_string_attribute(
        parameter, "payload.countermeasure.jitter.loop_iterations.provenance",
        "derived-exact");
    check(H5Dclose(parameter), "close jitter parameter");
  }
  check(H5Fclose(file), "close file");
}

class CapturingProgressSink final : public leakflow::ProgressSink {
public:
  void report(leakflow::Element &,
              const leakflow::ElementProgress &progress) override {
    reports.push_back(progress);
  }

  std::vector<leakflow::ElementProgress> reports;
};

} // namespace

int main() {
  namespace plugin = leakflow::plugins::extras;
  const auto sync_path = temp_path("sync");
  const auto jitter_path = temp_path("jitter");
  const auto invalid_alignment_path = temp_path("invalid_alignment");
  make_fixture(sync_path, false);
  make_fixture(jitter_path, true);
  make_fixture(invalid_alignment_path, false, false);

  plugin::Hdf5FileSrc source;
  source.set_property("path", sync_path.string());
  source.set_property("row_start", std::int64_t{1});
  source.set_property("row_count", std::int64_t{3});
  source.set_property("io_batch_rows", std::int64_t{1});
  CapturingProgressSink load_progress;
  source.set_progress_sink(&load_progress);
  const auto outputs = source.process_pads({});
  if (!expect(outputs.size() == 4 && !outputs.contains("countermeasures"),
              "sync HDF5 source output pads were wrong")) {
    return 1;
  }
  const auto traces =
      outputs.at("traces").payload_as<leakflow::base::TorchTensorPayload>();
  const auto keys =
      outputs.at("keys").payload_as<leakflow::base::TorchTensorPayload>();
  if (!expect(traces && traces->tensor().sizes().vec() ==
                            std::vector<std::int64_t>{3, 3},
              "HDF5 source selected trace shape was wrong")) {
    return 1;
  }
  if (!expect(traces->tensor().data_ptr<float>()[0] == 10.F &&
                  traces->tensor().data_ptr<float>()[8] == 32.F,
              "HDF5 source selected trace values were wrong")) {
    return 1;
  }
  if (!expect(keys &&
                  keys->tensor().sizes().vec() == std::vector<std::int64_t>{16},
              "HDF5 source sliced the fixed key")) {
    return 1;
  }
  if (!expect(
          outputs.at("traces").metadata("capture.source") == "ChipWhisperer" &&
              outputs.at("traces").metadata("tensor.axes") == "trace,sample" &&
              outputs.at("traces").metadata("payload.leakage.inverted") ==
                  "false" &&
              !outputs.at("traces").has_metadata("payload.leakage.polarity") &&
              !outputs.at("plaintexts")
                   .has_metadata("payload.leakage.inverted") &&
              !outputs.at("keys").has_metadata("payload.leakage.inverted") &&
              outputs.at("traces").metadata("origin.row.begin") == "1" &&
              outputs.at("traces").metadata("origin.row.count") == "3",
          "HDF5 source metadata was wrong")) {
    return 1;
  }
  // Tiny fixtures may finish inside the core's ~33 ms progress throttle, so
  // intermediate chunk reports can be coalesced. The reader test separately
  // proves one callback per hyperslab; the element must always expose start
  // and exact completion through the observer-facing progress channel.
  if (!expect(load_progress.reports.size() >= 2 &&
                  load_progress.reports.front().fraction == 0.0 &&
                  load_progress.reports.back().fraction == 1.0,
              "HDF5 source did not report chunked load progress")) {
    return 1;
  }

  plugin::Hdf5FileSrc jitter_source;
  jitter_source.set_property("path", jitter_path.string());
  const auto jitter_outputs = jitter_source.process_pads({});
  const auto bundle =
      jitter_outputs.at("countermeasures")
          .payload_as<leakflow::base::TorchTensorBundlePayload>();
  if (!expect(bundle && bundle->has("jitter.parameters.loop_iterations"),
              "HDF5 source countermeasure bundle was missing jitter labels")) {
    return 1;
  }
  if (!expect(
          bundle->tensor("jitter.parameters.loop_iterations").sizes().vec() ==
                  std::vector<std::int64_t>{5} &&
              jitter_outputs.at("countermeasures")
                      .metadata("payload.countermeasure.jitter.loop_iterations."
                                "provenance") == "derived-exact" &&
              !jitter_outputs.at("traces").has_metadata(
                  "payload.countermeasure.jitter.loop_iterations."
                  "provenance") &&
              jitter_outputs.at("countermeasures")
                      .metadata("payload.countermeasure.dims") ==
                  "jitter.parameters.loop_iterations=trace",
          "HDF5 source countermeasure dimensions were wrong")) {
    return 1;
  }

  plugin::Hdf5FileSrc invalid_alignment_source;
  invalid_alignment_source.set_property("path",
                                        invalid_alignment_path.string());
  if (!expect(
          throws_exception<std::invalid_argument>(
              [&] { (void)invalid_alignment_source.process_pads({}); }),
          "HDF5 source accepted an invalid traces row-alignment contract")) {
    return 1;
  }

  plugin::FakeLiveHdf5Src live;
  live.set_property("path", jitter_path.string());
  live.set_property("row_start", std::int64_t{1});
  live.set_property("row_count", std::int64_t{3});
  live.set_property("batch_size", std::int64_t{2});
  live.set_property("io_batch_rows", std::int64_t{1});
  CapturingProgressSink live_progress;
  live.set_progress_sink(&live_progress);
  live.start();
  auto first = live.process_pads({});
  if (!expect(!live.at_end_of_stream() &&
                  first.at("traces").metadata("origin.row.begin") == "1" &&
                  first.at("traces").metadata("origin.row.count") == "2",
              "fake-live first HDF5 batch metadata was wrong")) {
    return 1;
  }
  const auto first_labels =
      first.at("countermeasures")
          .payload_as<leakflow::base::TorchTensorBundlePayload>();
  if (!expect(first_labels &&
                  first_labels->tensor("jitter.parameters.loop_iterations")
                          .sizes()
                          .vec() == std::vector<std::int64_t>{2} &&
                  first_labels->tensor("jitter.parameters.loop_iterations")
                          .data_ptr<std::uint8_t>()[0] == 1,
              "fake-live rank-one countermeasure batch was wrong")) {
    return 1;
  }
  auto second = live.process_pads({});
  const auto second_traces =
      second.at("traces").payload_as<leakflow::base::TorchTensorPayload>();
  if (!expect(live.at_end_of_stream() && second_traces &&
                  second_traces->tensor().size(0) == 1,
              "fake-live final HDF5 batch or EOS was wrong")) {
    return 1;
  }
  if (!expect(live_progress.reports.front().fraction == 0.0 &&
                  live_progress.reports.back().fraction == 1.0 &&
                  live_progress.reports.back().index == 3 &&
                  live_progress.reports.back().total == 3,
              "fake-live HDF5 progress was wrong")) {
    return 1;
  }
  live.stop();
  live.start();
  if (!expect(!live.at_end_of_stream(),
              "fake-live HDF5 source did not restart")) {
    return 1;
  }
  live.stop();

  std::filesystem::remove(sync_path);
  std::filesystem::remove(jitter_path);
  std::filesystem::remove(invalid_alignment_path);
  return 0;
}
