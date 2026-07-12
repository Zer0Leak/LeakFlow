#include "leakflow/plugins/extras/hdf5_file_src.hpp"

#include "hdf5_source_common.hpp"
#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/extras/hdf5_tensor_dataset_reader.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow::plugins::extras {
namespace {

[[nodiscard]] PropertyEffect data_source_effect() {
  return {
      .kind = PropertyEffectKind::PayloadOutput,
      .scope = PropertyInvalidationScope::Downstream,
      .output_pads = {"traces", "plaintexts", "keys", "ciphertexts",
                      "countermeasures"},
  };
}

[[nodiscard]] Pad optional_torch_output(std::string name) {
  return Pad(std::move(name), PadDirection::Output,
             Caps(leakflow::base::torch_tensor_caps_type),
             PadPresence::Optional);
}

} // namespace

ElementDescriptor Hdf5FileSrc::descriptor() {
  return {
      .type_name = "Hdf5FileSrc",
      .long_name = "HDF5 Tensor Dataset Source",
      .rank = ElementRank::Primary,
      .klass = "Source/File/HDF5",
      .purpose = "load a LeakFlow HDF5 tensor dataset into aligned complete "
                 "Torch payloads",
      .output_pads =
          {
              optional_torch_output("traces"),
              optional_torch_output("plaintexts"),
              optional_torch_output("keys"),
              optional_torch_output("ciphertexts"),
              Pad("countermeasures", PadDirection::Output,
                  Caps("leakflow/torch-tensor-bundle"), PadPresence::Optional),
          },
      .property_specs =
          {
              PropertySpec("path", std::string(),
                           "path to a LeakFlow tensor-dataset HDF5 file", "",
                           std::monostate{}, "*.h5 or *.hdf5",
                           data_source_effect()),
              PropertySpec(
                  "device", std::string("cpu"),
                  "target Torch device after the HDF5 tensors are read on CPU",
                  "", std::monostate{},
                  "a Torch device string such as cpu, cuda, or cuda:0",
                  data_source_effect()),
              PropertySpec("row_start", std::int64_t{0},
                           "first row to load from row-aligned arrays", "row",
                           IntRangeConstraint{
                               0, std::numeric_limits<std::int64_t>::max()},
                           "", data_source_effect()),
              PropertySpec(
                  "row_count", std::int64_t{0},
                  "number of rows to load; 0 loads every remaining row", "rows",
                  IntRangeConstraint{0,
                                     std::numeric_limits<std::int64_t>::max()},
                  "", data_source_effect()),
              PropertySpec("io_batch_rows", std::int64_t{256},
                           "rows per internal HDF5 hyperslab read; output "
                           "remains one complete tensor per pad",
                           "rows",
                           IntRangeConstraint{
                               1, std::numeric_limits<std::int64_t>::max()},
                           "", PropertyEffect{}),
          },
      .telemetry_specs =
          {
              make_duration_telemetry_spec(
                  "storage_read",
                  "time spent materializing outputs after HDF5 dataset "
                  "inspection, including reads and device transfer"),
          },
      .keywords = {"hdf5", "h5", "tensor", "dataset", "source", "extras",
                   "torch", "sca"},
      .metadata_set_by_element =
          {
              make_element_metadata_descriptor("origin.file.format",
                                               std::string(), "storage format",
                                               {"hdf5"}),
              make_element_metadata_descriptor(
                  "origin.file.path", std::string(), "input HDF5 path",
                  {"traces/aes/sync/aes_sync_poi/key_01.h5"}),
              make_element_metadata_descriptor("origin.file.size",
                                               std::int64_t{},
                                               "input HDF5 size in bytes"),
              make_element_metadata_descriptor(
                  "origin.hdf5.dataset", std::string(),
                  "HDF5 dataset path represented by this output",
                  {"/traces", "/plaintexts", "/keys", "/ciphertexts",
                   "/countermeasures"}),
              make_element_metadata_descriptor(
                  "origin.role", std::string(), "semantic role of this output",
                  {"traces", "plaintexts", "keys", "ciphertexts",
                   "countermeasures"}),
              make_element_metadata_descriptor(
                  "origin.row.begin", std::int64_t{}, "first selected row"),
              make_element_metadata_descriptor(
                  "origin.row.count", std::int64_t{}, "selected row count"),
              make_element_metadata_descriptor(
                  "origin.row.total", std::int64_t{}, "file row count"),
              make_element_metadata_descriptor(
                  "tensor.axes", std::string(), "semantic Torch tensor axes",
                  {"trace,sample", "trace,byte", "key_byte"}),
              make_element_metadata_descriptor(
                  "payload.countermeasure.dims", std::string(),
                  "semantic axes for named countermeasure tensors",
                  {"jitter.parameters.loop_iterations=trace"}),
              make_element_metadata_descriptor(
                  "payload.countermeasure.tensors", std::string(),
                  "comma-separated tensor names in the countermeasure bundle",
                  {"jitter.parameters.loop_iterations"}),
          },
      .metadata_suggestions =
          {
              make_element_metadata_descriptor(
                  "capture.dataset.name", std::string(), "dataset identifier",
                  {"aes_sync_poi", "aes_jitter_attack"}),
              make_element_metadata_descriptor(
                  "capture.source", std::string(),
                  "capture hardware or acquisition source", {"ChipWhisperer"}),
              make_element_metadata_descriptor("capture.sample_rate_hz", 0.0,
                                               "trace sample rate",
                                               {"29454545.454545453"}),
          },
  };
}

Hdf5FileSrc::Hdf5FileSrc(std::string name) : Element(std::move(name)) {
  configure_from_descriptor(descriptor());
}

std::optional<Buffer> Hdf5FileSrc::process(std::optional<Buffer>) {
  throw std::invalid_argument(
      "Hdf5FileSrc emits named output pads; use process_pads");
}

ElementOutputs Hdf5FileSrc::process_pads(ElementInputs inputs) {
  if (!inputs.empty()) {
    throw std::invalid_argument(
        "Hdf5FileSrc is a source and does not accept input buffers");
  }

  const auto options = detail::source_options(*this);
  report_progress(0.0, "Opening HDF5 tensor dataset", 0, 0);
  leakflow::extras::Hdf5TensorDatasetReader reader(options.path);

  // Bail before reading if a stop is already pending (mirrors GaussianMixture).
  if (!cooperative_checkpoint()) {
    report_progress(1.0, "cancelled", 0, 0, leakflow::ProgressStatus::Cancelled);
    return {};
  }

  std::uint64_t total_bytes = 0;
  ElementOutputs outputs;
  try {
    auto storage_scope = profile_scope("storage_read");
    outputs = detail::read_hdf5_outputs(
        reader, options,
        [this, &total_bytes](const detail::AggregateReadProgress &update) {
          total_bytes = update.total_logical_bytes;
          if (!cooperative_checkpoint()) {
            return false;
          }
          const auto fraction =
              update.total_logical_bytes == 0
                  ? 0.0
                  : 0.99 * static_cast<double>(update.logical_bytes_read) /
                        static_cast<double>(update.total_logical_bytes);
          report_progress(fraction,
                          "Reading " + update.array_path + ": " +
                              std::to_string(update.rows_read) + "/" +
                              std::to_string(update.total_rows) + " rows",
                          update.logical_bytes_read,
                          update.total_logical_bytes);
          // Park while paused; abort the read on stop (unwinds the reader's
          // hyperslab loop as TensorReadCancelled).
          return cooperative_checkpoint();
        });
  } catch (const leakflow::extras::TensorReadCancelled &) {
    report_progress(1.0, "cancelled", total_bytes, total_bytes,
                    leakflow::ProgressStatus::Cancelled);
    return {};
  }
  if (!cooperative_checkpoint()) {
    report_progress(1.0, "cancelled", total_bytes, total_bytes,
                    leakflow::ProgressStatus::Cancelled);
    return {};
  }
  report_progress(1.0, "HDF5 tensor dataset loaded", total_bytes, total_bytes,
                  leakflow::ProgressStatus::Completed);

  auto record = make_log_record(log::LogLevel::Debug, "element",
                                "loaded HDF5 tensor dataset");
  record.fields.emplace("origin.file.path", options.path.string());
  record.fields.emplace("output_pads", std::to_string(outputs.size()));
  leakflow::log::write(std::move(record));
  return outputs;
}

} // namespace leakflow::plugins::extras
