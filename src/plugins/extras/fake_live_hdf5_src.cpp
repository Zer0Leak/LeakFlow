#include "leakflow/plugins/extras/fake_live_hdf5_src.hpp"

#include "hdf5_source_common.hpp"
#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/extras/hdf5_tensor_dataset_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace leakflow::plugins::extras {
namespace {

[[nodiscard]] Pad optional_torch_output(std::string name) {
  return Pad(std::move(name), PadDirection::Output,
             Caps(leakflow::base::torch_tensor_caps_type),
             PadPresence::Optional);
}

[[nodiscard]] PropertyEffect lifecycle_effect() {
  return {
      .kind = PropertyEffectKind::Lifecycle,
      .scope = PropertyInvalidationScope::FullPipeline,
  };
}

[[nodiscard]] std::int64_t integer_property_or(const Element &element,
                                               std::string_view name,
                                               std::int64_t fallback) {
  if (const auto value = element.property_as<std::int64_t>(name)) {
    return *value;
  }
  return fallback;
}

[[nodiscard]] double double_property_or(const Element &element,
                                        std::string_view name,
                                        double fallback) {
  if (const auto value = element.property_as<double>(name)) {
    return *value;
  }
  return fallback;
}

// Paces the replay to trace_rate, cooperating with pause/stop through the
// supplied checkpoint (Element::cooperative_checkpoint): it parks while paused
// and returns false on stop, so a paused live source freezes between batches
// and a stopped one unwinds the pacing sleep promptly.
[[nodiscard]] bool pace(const std::function<bool()> &checkpoint,
                        double trace_rate, std::uint64_t rows) {
  if (trace_rate <= 0.0 || rows == 0) {
    return true;
  }

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration<double>(static_cast<double>(rows) / trace_rate);
  constexpr auto poll = std::chrono::milliseconds(10);
  while (true) {
    if (!checkpoint()) {
      return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return true;
    }
    std::this_thread::sleep_for(std::min(
        poll,
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now) +
            std::chrono::milliseconds(1)));
  }
}

} // namespace

struct FakeLiveHdf5Src::Impl {
  detail::Hdf5SourceOptions base_options;
  std::unique_ptr<leakflow::extras::Hdf5TensorDatasetReader> reader;
  std::uint64_t total_rows = 0;
  std::uint64_t emitted_rows = 0;
  bool terminal_progress_reported = false;
};

ElementDescriptor FakeLiveHdf5Src::descriptor() {
  return {
      .type_name = "FakeLiveHdf5Src",
      .long_name = "Fake Live HDF5 Tensor Dataset Source",
      .rank = ElementRank::Primary,
      .klass = "Source/Live/HDF5",
      .purpose =
          "replay aligned HDF5 tensor rows as paced live Torch buffer batches",
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
                           lifecycle_effect()),
              PropertySpec("device", std::string("cpu"),
                           "target Torch device for emitted batches", "",
                           std::monostate{},
                           "a Torch device string such as cpu, cuda, or cuda:0",
                           lifecycle_effect()),
              PropertySpec("row_start", std::int64_t{0}, "first row to replay",
                           "row",
                           IntRangeConstraint{
                               0, std::numeric_limits<std::int64_t>::max()},
                           "", lifecycle_effect()),
              PropertySpec(
                  "row_count", std::int64_t{0},
                  "number of rows to replay; 0 uses every remaining row",
                  "rows",
                  IntRangeConstraint{0,
                                     std::numeric_limits<std::int64_t>::max()},
                  "", lifecycle_effect()),
              PropertySpec(
                  "batch_size", std::int64_t{1},
                  "trace rows emitted together in one live buffer batch",
                  "traces",
                  IntRangeConstraint{1,
                                     std::numeric_limits<std::int64_t>::max()},
                  "", lifecycle_effect()),
              PropertySpec("io_batch_rows", std::int64_t{256},
                           "rows per internal HDF5 hyperslab read", "rows",
                           IntRangeConstraint{
                               1, std::numeric_limits<std::int64_t>::max()},
                           "", PropertyEffect{}),
              PropertySpec(
                  "trace_rate", 0.0,
                  "replay rate in traces per second; 0 disables pacing",
                  "traces/s", DoubleRangeConstraint{0.0, 1.0e15}, "",
                  PropertyEffect{}),
          },
      .telemetry_specs =
          {
              make_duration_telemetry_spec(
                  "storage_read",
                  "time spent materializing each output batch after HDF5 "
                  "dataset inspection, including reads and device transfer"),
          },
      .keywords = {"hdf5", "h5", "live", "fake", "replay", "dataset", "torch",
                   "sca"},
      .metadata_set_by_element =
          {
              make_element_metadata_descriptor("origin.file.format",
                                               std::string(), "storage format",
                                               {"hdf5"}),
              make_element_metadata_descriptor(
                  "origin.file.path", std::string(), "input HDF5 path"),
              make_element_metadata_descriptor("origin.file.size",
                                               std::int64_t{},
                                               "input HDF5 size in bytes"),
              make_element_metadata_descriptor("origin.hdf5.dataset",
                                               std::string(),
                                               "source HDF5 dataset path"),
              make_element_metadata_descriptor("origin.role", std::string(),
                                               "semantic output role"),
              make_element_metadata_descriptor("origin.row.begin",
                                               std::int64_t{},
                                               "first row in this batch"),
              make_element_metadata_descriptor(
                  "origin.row.count", std::int64_t{}, "rows in this batch"),
              make_element_metadata_descriptor(
                  "origin.row.total", std::int64_t{}, "file row count"),
              make_element_metadata_descriptor(
                  "tensor.axes", std::string(), "semantic Torch tensor axes",
                  {"trace,sample", "trace,byte", "key_byte"}),
              make_element_metadata_descriptor(
                  "payload.layout", std::string(),
                  "ordered semantic axes of the emitted payload",
                  {"trace/sample", "trace/byte", "key_byte",
                   "jitter.parameters.loop_iterations=trace"}),
              make_element_metadata_descriptor(
                  "payload.countermeasure.tensors", std::string(),
                  "comma-separated tensor names in the countermeasure bundle",
                  {"jitter.parameters.loop_iterations"}),
          },
      .can_replay = false,
      .live_source = true,
  };
}

FakeLiveHdf5Src::FakeLiveHdf5Src(std::string name)
    : Element(std::move(name)), impl_(std::make_unique<Impl>()) {
  configure_from_descriptor(descriptor());
}

FakeLiveHdf5Src::~FakeLiveHdf5Src() = default;

void FakeLiveHdf5Src::start() {
  impl_->base_options = detail::source_options(*this);
  impl_->reader = std::make_unique<leakflow::extras::Hdf5TensorDatasetReader>(
      impl_->base_options.path);
  impl_->total_rows = detail::selected_row_count(impl_->reader->descriptor(),
                                                 impl_->base_options.row_start,
                                                 impl_->base_options.row_count);
  impl_->emitted_rows = 0;
  if (impl_->total_rows == 0) {
    report_progress(1.0, "HDF5 live replay is empty", 0, 0);
    impl_->terminal_progress_reported = true;
  } else {
    report_progress(0.0, "Starting HDF5 live replay", 0, impl_->total_rows);
  }
}

std::optional<Buffer> FakeLiveHdf5Src::process(std::optional<Buffer>) {
  throw std::invalid_argument(
      "FakeLiveHdf5Src emits named output pads; use process_pads");
}

ElementOutputs FakeLiveHdf5Src::process_pads(ElementInputs inputs) {
  if (!inputs.empty()) {
    throw std::invalid_argument(
        "FakeLiveHdf5Src is a source and does not accept input buffers");
  }
  if (!impl_->reader) {
    throw std::logic_error("FakeLiveHdf5Src must be started before processing");
  }
  if (at_end_of_stream()) {
    return {};
  }

  const auto configured_batch =
      static_cast<std::uint64_t>(integer_property_or(*this, "batch_size", 1));
  const auto batch_rows =
      std::min(configured_batch, impl_->total_rows - impl_->emitted_rows);
  auto options = impl_->base_options;
  const auto configured_io_rows =
      integer_property_or(*this, "io_batch_rows", 256);
  if (configured_io_rows <= 0) {
    throw std::invalid_argument(
        "FakeLiveHdf5Src io_batch_rows must be greater than zero");
  }
  options.io_batch_rows = static_cast<std::uint64_t>(configured_io_rows);
  options.row_start += impl_->emitted_rows;
  options.row_count = batch_rows;

  ElementOutputs outputs;
  try {
    auto storage_scope = profile_scope("storage_read");
    // Live semantics: abort a batch read on stop, but do not park on pause
    // mid-batch (pausing a live source is handled at the between-batch safe
    // point). Hence a stop-only check here, not cooperative_checkpoint.
    outputs = detail::read_hdf5_outputs(
        *impl_->reader, options,
        [this](const detail::AggregateReadProgress &) {
          return !stop_token().stop_requested();
        });
  } catch (const leakflow::extras::TensorReadCancelled &) {
    const auto fraction = impl_->total_rows == 0
                              ? 1.0
                              : static_cast<double>(impl_->emitted_rows) /
                                    static_cast<double>(impl_->total_rows);
    report_progress(fraction, "cancelled", impl_->emitted_rows,
                    impl_->total_rows, leakflow::ProgressStatus::Cancelled);
    impl_->terminal_progress_reported = true;
    return {};
  }

  const auto report_cancelled = [this]() {
    report_progress(1.0, "cancelled", impl_->emitted_rows,
                    impl_->total_rows, leakflow::ProgressStatus::Cancelled);
    impl_->terminal_progress_reported = true;
  };
  if (!pace([this] { return cooperative_checkpoint(); },
            double_property_or(*this, "trace_rate", 0.0), batch_rows) ||
      !cooperative_checkpoint()) {
    report_cancelled();
    return {};
  }

  impl_->emitted_rows += batch_rows;
  const auto fraction = impl_->total_rows == 0
                            ? 1.0
                            : static_cast<double>(impl_->emitted_rows) /
                                  static_cast<double>(impl_->total_rows);
  const auto status = impl_->emitted_rows >= impl_->total_rows
                          ? leakflow::ProgressStatus::Completed
                          : leakflow::ProgressStatus::Active;
  report_progress(
      fraction,
      "Replaying HDF5 traces: " + std::to_string(impl_->emitted_rows) + "/" +
          std::to_string(impl_->total_rows),
      impl_->emitted_rows, impl_->total_rows, status);
  if (status == leakflow::ProgressStatus::Completed) {
    impl_->terminal_progress_reported = true;
  }

  auto record = make_log_record(log::LogLevel::Debug, "element",
                                "emitted HDF5 live batch");
  record.fields.emplace("origin.file.path", impl_->base_options.path.string());
  record.fields.emplace("origin.row.count", std::to_string(batch_rows));
  record.fields.emplace("origin.row.emitted",
                        std::to_string(impl_->emitted_rows));
  leakflow::log::write(std::move(record));
  return outputs;
}

bool FakeLiveHdf5Src::at_end_of_stream() const {
  return impl_->reader && impl_->emitted_rows >= impl_->total_rows;
}

bool FakeLiveHdf5Src::can_replay() const { return false; }

void FakeLiveHdf5Src::stop() {
  if (impl_->reader && impl_->emitted_rows < impl_->total_rows &&
      !impl_->terminal_progress_reported) {
    report_progress(1.0, "cancelled", impl_->emitted_rows,
                    impl_->total_rows, leakflow::ProgressStatus::Cancelled);
  }
  impl_ = std::make_unique<Impl>();
}

} // namespace leakflow::plugins::extras
