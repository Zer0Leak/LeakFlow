#include "leakflow/plugins/base/fake_live_src.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace leakflow::plugins::base {
namespace {

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] double double_property_or(const Element& element, std::string_view name, double fallback)
{
    if (const auto value = element.property_as<double>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] torch::Tensor load_pickle_tensor(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("FakeLiveSrc could not open input path: " + path);
    }
    std::vector<char> data{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return torch::pickle_load(data).toTensor();
}

[[nodiscard]] Caps torch_tensor_caps()
{
    return Caps(leakflow::base::torch_tensor_caps_type);
}

} // namespace

ElementDescriptor FakeLiveSrc::descriptor()
{
    return {
        .type_name = "FakeLiveSrc",
        .klass = "Source/Live/Torch",
        .purpose = "simulate a live capture: stream one Buffer per axis-0 row of a Torch .pt tensor",
        .output_pads =
            {
                Pad("src", PadDirection::Output, torch_tensor_caps()),
            },
        .property_specs =
            {
                PropertySpec("path", std::string(), "path to the Torch .pt tensor streamed row-by-row"),
                PropertySpec("trace_rate", 0.0,
                             "if > 0: pace the fake live stream to one trace per 1/rate seconds "
                             "(rate = traces emitted per second); 0 = no delay",
                             "traces/s", DoubleRangeConstraint{0.0, 1.0e15}, ""),
            },
        .keywords = {"live", "stream", "source", "fake", "capture"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "origin.file.path",
                std::string(),
                "input Torch tensor file path streamed row-by-row",
                {"traces/aes/sync/aes_sync_poi/key_01/traces.pt"}),
            make_element_metadata_descriptor(
                "origin.row_index",
                std::int64_t{},
                "axis-0 row index of the streamed buffer",
                {"0", "1", "2"}),
        },
        .metadata_suggestions = {
            make_element_metadata_descriptor(
                "capture.sample_rate_hz",
                0.0,
                "trace sample rate in samples per second",
                {"29454545.454545453"},
                "TracePlot uses this user metadata when x_axis=time_us and the sample_rate_hz property is zero."),
            make_element_metadata_descriptor(
                "capture.source",
                std::string(),
                "capture hardware, simulator, or acquisition source",
                {"ChipWhisperer"},
                "Useful for plot titles, reports, and reproducibility metadata. FakeLiveSrc does not "
                "set it automatically."),
            make_element_metadata_descriptor(
                "capture.dataset.name",
                std::string(),
                "dataset or experiment identifier for downstream reporting",
                {"aes_sync_poi"}),
            make_element_metadata_descriptor(
                "payload.leakage.range",
                DoubleInterval{0.0, 0.0},
                "expected leakage value range",
                {"[-0.5,0.5]"},
                "Plotters and reports may use this to choose consistent y-axis bounds.",
                "[min,max]"),
            make_element_metadata_descriptor(
                "payload.leakage.inverted",
                false,
                "whether the streamed leakage samples are inverted, as declared by the user",
                {"false", "true"},
                "FakeLiveSrc does not set this automatically; the user records whether the streamed "
                "leakage samples are inverted.",
                "true or false"),
        },
        .live_source = true,
    };
}

FakeLiveSrc::FakeLiveSrc(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

void FakeLiveSrc::ensure_loaded()
{
    if (loaded_) {
        return;
    }
    const auto path = string_property_or(*this, "path", "");
    if (path.empty()) {
        throw std::invalid_argument("FakeLiveSrc path property must not be empty");
    }
    tensor_ = load_pickle_tensor(path);
    if (!tensor_.defined() || tensor_.dim() < 1) {
        throw std::invalid_argument("FakeLiveSrc tensor must have at least one dimension");
    }
    cursor_ = 0;
    loaded_ = true;
}

std::optional<Buffer> FakeLiveSrc::process(std::optional<Buffer>)
{
    ensure_loaded();
    if (cursor_ >= tensor_.size(0)) {
        return std::nullopt; // exhausted; the pump stops first via at_end_of_stream()
    }

    auto row = tensor_.narrow(0, cursor_, 1).contiguous(); // one axis-0 entry, kept rank-2 [1, ...]
    const auto row_index = cursor_;
    ++cursor_;

    auto payload = leakflow::base::TorchTensorPayload(std::move(row));
    Buffer buffer(payload.caps());
    // capture.source is left to the user as suggested metadata (like TorchFileSrc):
    // FakeLiveSrc does not assert a hardware/source identity for replayed data.
    buffer.set_metadata("origin.file.path", string_property_or(*this, "path", ""));
    buffer.set_metadata("origin.row_index", std::to_string(row_index));

    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "streamed live row");
    record.fields.emplace("origin.row_index", std::to_string(row_index));
    leakflow::log::write(std::move(record));

    // Pace the stream to one buffer (trace) per 1/trace_rate seconds. This is
    // intentionally separate from capture.sample_rate_hz metadata: capture sample
    // rate describes samples inside a trace, while trace_rate describes fake-live
    // replay speed. 0 = no delay. The wait polls the cooperative-stop token (S11.8)
    // so Ctrl+C / window-close interrupts it mid-trace instead of running it to
    // completion.
    const auto trace_rate = double_property_or(*this, "trace_rate", 0.0);
    if (trace_rate > 0.0) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::duration<double>(1.0 / trace_rate);
        const auto& token = stop_token();
        constexpr auto poll = std::chrono::milliseconds(10);
        while (!token.stop_requested()) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                break;
            }
            std::this_thread::sleep_for(std::min(poll, std::chrono::duration_cast<std::chrono::milliseconds>(
                                                          deadline - now) + std::chrono::milliseconds(1)));
        }
    }

    return buffer;
}

bool FakeLiveSrc::at_end_of_stream() const
{
    return loaded_ && cursor_ >= tensor_.size(0);
}

void FakeLiveSrc::stop()
{
    // Reset so a fresh start_all -> run -> stop_all cycle replays the stream.
    cursor_ = 0;
    loaded_ = false;
}

} // namespace leakflow::plugins::base
