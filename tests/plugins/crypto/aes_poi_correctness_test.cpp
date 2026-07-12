#include "leakflow/base/plot_annotation_payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/crypto/aes.hpp"
#include "leakflow/plugins/crypto/crypto_elements.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace {

namespace crypto_plugin = leakflow::plugins::crypto;

// A real AES first-round S-box leak in a low-noise synchronized capture
// correlates strongly at its point of interest. With 50 traces the random
// correlation noise floor is roughly 1/sqrt(50) ~= 0.14, so a true PoI must
// clear this with comfortable margin.
constexpr double poi_correlation_threshold = 0.3;
constexpr double numeric_tolerance = 1.0e-6;

bool expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

bool expect_near(double actual, double expected, const std::string& message)
{
    if (std::abs(actual - expected) > numeric_tolerance) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
        return false;
    }

    return true;
}

torch::Tensor load_pickle_tensor(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open fixture: " + path.string());
    }

    std::vector<char> data{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return torch::pickle_load(data).toTensor();
}

leakflow::Buffer torch_buffer(torch::Tensor tensor)
{
    auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    leakflow::Buffer buffer(payload.caps());
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    return buffer;
}

// Reference HW(m)/HW(y) computed straight from the fixture bytes, using the
// raw S-box table and std::popcount, independently of the tensorized leakage
// path inside AesLeakage.
struct ReferenceLeakage {
    torch::Tensor hw_m; // [N] uint8
    torch::Tensor hw_y; // [N] uint8
};

ReferenceLeakage reference_leakage(
    const torch::Tensor& plaintexts,
    const torch::Tensor& key,
    std::size_t byte_index)
{
    const auto trace_count = plaintexts.size(0);
    const auto plaintexts_cpu = plaintexts.to(torch::kCPU).contiguous();
    const auto key_cpu = key.to(torch::kCPU).contiguous();
    const auto plaintext_access = plaintexts_cpu.accessor<std::uint8_t, 2>();
    const auto key_access = key_cpu.accessor<std::uint8_t, 1>();
    const auto key_byte = key_access[static_cast<std::int64_t>(byte_index)];

    auto hw_m = torch::empty({trace_count}, torch::TensorOptions().dtype(torch::kUInt8));
    auto hw_y = torch::empty({trace_count}, torch::TensorOptions().dtype(torch::kUInt8));
    auto hw_m_access = hw_m.accessor<std::uint8_t, 1>();
    auto hw_y_access = hw_y.accessor<std::uint8_t, 1>();

    for (std::int64_t trace = 0; trace < trace_count; ++trace) {
        const auto m = plaintext_access[trace][static_cast<std::int64_t>(byte_index)];
        const auto y = leakflow::crypto::aes::sbox_table[static_cast<std::size_t>(m ^ key_byte)];
        hw_m_access[trace] = static_cast<std::uint8_t>(std::popcount(static_cast<std::uint8_t>(m)));
        hw_y_access[trace] = static_cast<std::uint8_t>(std::popcount(static_cast<std::uint8_t>(y)));
    }

    return ReferenceLeakage{.hw_m = std::move(hw_m), .hw_y = std::move(hw_y)};
}

// Reference Pearson correlation between every trace sample and the target
// vector, computed in float64 directly from the fixtures.
struct ReferencePoi {
    std::int64_t sample_index = 0;
    double correlation = 0.0;
    torch::Tensor curve; // [M] float64
};

ReferencePoi reference_poi(const torch::Tensor& traces, const torch::Tensor& target)
{
    const auto features = traces.to(torch::kFloat64);
    const auto target_vector = target.to(torch::kFloat64);

    const auto centered_features = features - features.mean(0, true);
    const auto centered_target = target_vector - target_vector.mean();
    const auto numerator = torch::matmul(centered_target.unsqueeze(0), centered_features).squeeze(0);
    const auto denominator =
        torch::sqrt(centered_target.square().sum() * centered_features.square().sum(0));
    const auto curve = (numerator / denominator).contiguous();

    const auto best = curve.abs().argmax().item<std::int64_t>();

    return ReferencePoi{
        .sample_index = best,
        .correlation = curve[best].item<double>(),
        .curve = curve,
    };
}

std::string format_fixed(double value, int precision)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    return output.str();
}

bool run_fixture(const std::string& label, const std::filesystem::path& fixture_dir, std::size_t byte_index)
{
    const auto traces = load_pickle_tensor(fixture_dir / "traces_first_50.pt");
    const auto plaintexts = load_pickle_tensor(fixture_dir / "plain_texts_first_50.pt");
    const auto key = load_pickle_tensor(fixture_dir / "key_first_50.pt");

    if (!expect(traces.dim() == 2 && plaintexts.dim() == 2 && plaintexts.size(1) == 16 && key.dim() == 1
                && key.size(0) == 16 && traces.size(0) == plaintexts.size(0),
            label + ": fixture shapes were unexpected")) {
        return false;
    }

    const auto reference = reference_leakage(plaintexts, key, byte_index);

    // ---- AesLeakage numeric correctness: HW(m) and HW(y) for a known byte ----
    crypto_plugin::AesLeakage leakage;
    leakage.set_property("byte_indexes", leakflow::IntList{static_cast<std::int64_t>(byte_index)});
    leakage.set_property(
        "channels",
        leakflow::StringList{
            crypto_plugin::aes_leakage_channel_hw_m,
            crypto_plugin::aes_leakage_channel_hw_y,
        });

    leakflow::ElementInputs leakage_inputs;
    leakage_inputs.emplace("traces", torch_buffer(traces));
    leakage_inputs.emplace("plaintexts", torch_buffer(plaintexts));
    leakage_inputs.emplace("keys", torch_buffer(key));
    const auto leakage_output = leakage.process_inputs(std::move(leakage_inputs));
    if (!expect(leakage_output.has_value(), label + ": AesLeakage produced no output")) {
        return false;
    }

    const auto leakage_payload = leakage_output->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(leakage_payload != nullptr, label + ": AesLeakage payload type was wrong")) {
        return false;
    }
    const auto& leakage_tensor = leakage_payload->tensor(); // [1, N, 2]
    if (!expect(leakage_tensor.dim() == 3 && leakage_tensor.size(0) == 1
                && leakage_tensor.size(1) == traces.size(0) && leakage_tensor.size(2) == 2,
            label + ": AesLeakage output shape was wrong")) {
        return false;
    }

    const auto produced_hw_m = leakage_tensor.select(0, 0).select(1, 0).contiguous();
    const auto produced_hw_y = leakage_tensor.select(0, 0).select(1, 1).contiguous();
    if (!expect(torch::equal(produced_hw_m, reference.hw_m),
            label + ": AesLeakage HW(m) did not match values computed from the fixture")) {
        return false;
    }
    if (!expect(torch::equal(produced_hw_y, reference.hw_y),
            label + ": AesLeakage HW(y) did not match values computed from the fixture")) {
        return false;
    }

    // Spot-check trace 0 against fully hand-derived values.
    const auto plaintexts_cpu = plaintexts.to(torch::kCPU).contiguous();
    const auto key_cpu = key.to(torch::kCPU).contiguous();
    const auto m0 = plaintexts_cpu.accessor<std::uint8_t, 2>()[0][static_cast<std::int64_t>(byte_index)];
    const auto k0 = key_cpu.accessor<std::uint8_t, 1>()[static_cast<std::int64_t>(byte_index)];
    const auto y0 = leakflow::crypto::aes::sbox_table[static_cast<std::size_t>(m0 ^ k0)];
    if (!expect(produced_hw_m[0].item<std::int64_t>() == std::popcount(static_cast<std::uint8_t>(m0)),
            label + ": AesLeakage HW(m) trace 0 spot-check failed")) {
        return false;
    }
    if (!expect(produced_hw_y[0].item<std::int64_t>() == std::popcount(static_cast<std::uint8_t>(y0)),
            label + ": AesLeakage HW(y) trace 0 spot-check failed")) {
        return false;
    }

    // ---- PoiSelect numeric correctness against an independent recompute ----
    crypto_plugin::AesLeakage target_leakage;
    target_leakage.set_property("byte_indexes", leakflow::IntList{static_cast<std::int64_t>(byte_index)});
    target_leakage.set_property("channels", leakflow::StringList{crypto_plugin::aes_leakage_channel_hw_y});
    leakflow::ElementInputs target_inputs;
    target_inputs.emplace("traces", torch_buffer(traces));
    target_inputs.emplace("plaintexts", torch_buffer(plaintexts));
    target_inputs.emplace("keys", torch_buffer(key));
    auto target_buffer = target_leakage.process_inputs(std::move(target_inputs));
    if (!expect(target_buffer.has_value(), label + ": HW(y) target leakage produced no output")) {
        return false;
    }

    const auto poi_reference = reference_poi(traces, reference.hw_y);
    if (!expect(std::abs(poi_reference.correlation) >= poi_correlation_threshold,
            label + ": reference PoI correlation did not clear the threshold")) {
        return false;
    }

    // PearsonCorrelator (accumulation) + PoiSelect (top-k) -- the split of the old
    // PearsonPoiFinder; the end-to-end PoI must be numerically identical.
    crypto_plugin::PearsonCorrelator correlator;
    // Match the independent recompute dtype so we can assert tight equality.
    correlator.set_property("compute_dtype", std::string("float64"));
    crypto_plugin::PoiSelect finder;
    finder.set_property("top_k", leakflow::IntList{1});
    finder.set_property("rank_by", leakflow::StringList{"abs"});

    leakflow::ElementInputs finder_inputs;
    finder_inputs.emplace("features", torch_buffer(traces));
    finder_inputs.emplace("targets", *target_buffer);
    const auto correlation_output = correlator.process_inputs(std::move(finder_inputs));
    if (!expect(correlation_output.has_value(), label + ": PearsonCorrelator produced no output")) {
        return false;
    }
    const auto poi_output = finder.process(*correlation_output);
    if (!expect(poi_output.has_value(), label + ": PoiSelect produced no output")) {
        return false;
    }

    const auto poi_payload = poi_output->payload_as<crypto_plugin::CorrelationPoiPayload>();
    if (!expect(poi_payload != nullptr && poi_payload->unit_count() == 1,
            label + ": PoiSelect result count was wrong")) {
        return false;
    }
    const auto& result = poi_payload->result(0);
    if (!expect(result.unit_index == byte_index, label + ": PoiSelect reported wrong unit index")) {
        return false;
    }
    // result tensor is [channel, top_k, 2]: pair is (sample_index, correlation).
    if (!expect(result.result.dim() == 3 && result.result.size(0) == 1 && result.result.size(1) == 1
                && result.result.size(2) == 2,
            label + ": PoiSelect result shape was wrong")) {
        return false;
    }

    const auto poi_sample_index = static_cast<std::int64_t>(result.result[0][0][0].item<double>());
    const auto poi_correlation = result.result[0][0][1].item<double>();
    if (!expect(poi_sample_index == poi_reference.sample_index,
            label + ": PoiSelect PoI sample index did not match the strongest correlation sample")) {
        return false;
    }
    if (!expect_near(poi_correlation, poi_reference.correlation,
            label + ": PoiSelect PoI correlation did not match the independent recompute")) {
        return false;
    }
    if (!expect(std::abs(poi_correlation) >= poi_correlation_threshold,
            label + ": PoiSelect PoI correlation magnitude did not clear the threshold")) {
        return false;
    }

    // ---- CorrelationPoiToPlotAnnotations numeric/formatting correctness ----
    constexpr int precision = 4;
    crypto_plugin::CorrelationPoiToPlotAnnotations converter;
    converter.set_property("value_format", std::string("fixed"));
    converter.set_property("precision", std::int64_t{precision});
    const auto annotation_output = converter.process(*poi_output);
    if (!expect(annotation_output.has_value(), label + ": annotation converter produced no output")) {
        return false;
    }

    const auto annotations = annotation_output->payload_as<leakflow::base::PlotAnnotationPayload>();
    if (!expect(annotations != nullptr && annotations->annotation_count() == 1,
            label + ": annotation count was wrong")) {
        return false;
    }
    const auto& annotation = annotations->annotation(0);
    if (!expect(annotation.sample_index == poi_sample_index,
            label + ": annotation sample index did not match the PoI output")) {
        return false;
    }
    if (!expect(annotation.norm_value && std::abs(*annotation.norm_value - poi_correlation) <= numeric_tolerance,
            label + ": annotation norm_value did not match the PoI correlation")) {
        return false;
    }
    const auto expected_label = "unit_" + std::to_string(byte_index) + "." + crypto_plugin::aes_leakage_channel_hw_y;
    if (!expect(annotation.label == expected_label, label + ": annotation label was wrong")) {
        return false;
    }

    const auto expected_value_text = format_fixed(poi_correlation, precision);
    if (!expect(!annotation.fields.empty() && annotation.fields.back().first == "correlation"
                && annotation.fields.back().second == expected_value_text,
            label + ": annotation correlation field text did not match the formatted PoI value")) {
        return false;
    }
    if (!expect(annotation.text == expected_label + ": " + expected_value_text,
            label + ": annotation text did not match the formatted PoI value")) {
        return false;
    }

    return true;
}

} // namespace

int main()
{
    constexpr std::size_t byte_index = 0;

    if (!run_fixture("key_01", LEAKFLOW_AES_FIXTURE_DIR_KEY_01, byte_index)) {
        return 1;
    }
    if (!run_fixture("key_02", LEAKFLOW_AES_FIXTURE_DIR_KEY_02, byte_index)) {
        return 1;
    }

    return 0;
}
