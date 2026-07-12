#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/payload_codec.hpp"
#include "leakflow/plugins/base/descriptor_catalog.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"
#include "leakflow/plugins/crypto/descriptor_catalog.hpp"
#include "leakflow/plugins/crypto/poi_select.hpp"
#include "leakflow/plugins/extras/buffer_file_sink.hpp"
#include "leakflow/plugins/extras/buffer_file_src.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>

namespace {

namespace extras_plugin = leakflow::plugins::extras;
namespace crypto_plugin = leakflow::plugins::crypto;

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

template <typename Function>
bool throws(Function function)
{
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    auto codecs = std::make_shared<leakflow::PayloadCodecRegistry>();
    leakflow::plugins::base::register_payload_codecs(*codecs);
    crypto_plugin::register_payload_codecs(*codecs);

    const auto root = std::filesystem::temp_directory_path() / "leakflow_buffer_file_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    int status = 0;
    const auto fail = [&status] { status = 1; };

    // --- TorchTensorPayload round-trip (envelope + native tensor dataset) ---
    {
        auto tensor = torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3});
        auto payload = std::make_shared<leakflow::base::TorchTensorPayload>(tensor);
        leakflow::Buffer buffer(payload->caps());
        buffer.set_metadata("capture.source", "ChipWhisperer");
        buffer.set_metadata("payload.leakage.channels", "HW(m),HW(y)");
        buffer.set_payload(payload);
        const auto original_caps = buffer.caps().to_string();

        const auto file = (root / "tensor.h5").string();
        extras_plugin::BufferFileSink sink(codecs);
        sink.set_property("path", file);
        (void)sink.process(buffer);

        if (!expect(std::filesystem::exists(file), "BufferFileSink did not write the HDF5 file")) {
            fail();
        }

        extras_plugin::BufferFileSrc src(codecs);
        src.set_property("path", file);
        const auto loaded = src.process(std::nullopt);

        if (!expect(loaded.has_value(), "BufferFileSrc produced no buffer")) {
            return 1;
        }
        if (!expect(loaded->caps().to_string() == original_caps, "caps did not round-trip")) {
            fail();
        }
        if (!expect(loaded->metadata("capture.source") == "ChipWhisperer", "capture metadata did not round-trip")) {
            fail();
        }
        if (!expect(loaded->metadata("payload.leakage.channels") == "HW(m),HW(y)",
                "leakage metadata did not round-trip")) {
            fail();
        }
        const auto loaded_payload = loaded->payload_as<leakflow::base::TorchTensorPayload>();
        if (!expect(loaded_payload != nullptr && torch::equal(loaded_payload->tensor(), tensor),
                "tensor payload did not round-trip")) {
            fail();
        }
    }

    // --- CorrelationPoiPayload round-trip (crypto codec, variable-length results) ---
    {
        auto results = std::vector<crypto_plugin::CorrelationPoiResult>{
            crypto_plugin::CorrelationPoiResult{
                .unit = 3,
                .result = torch::tensor({{{7.0, 0.625}, {2.0, -0.5}}}, torch::TensorOptions().dtype(torch::kFloat64)),
            },
            crypto_plugin::CorrelationPoiResult{
                .unit = 5,
                .result = torch::tensor({{{1.0, 0.9}}}, torch::TensorOptions().dtype(torch::kFloat64)),
            },
        };
        auto payload = std::make_shared<crypto_plugin::CorrelationPoiPayload>(results, "correlation");
        leakflow::Buffer buffer{leakflow::Caps(crypto_plugin::correlation_poi_caps_type)};
        buffer.set_metadata("payload.poi.method", crypto_plugin::pearson_poi_method_id);
        buffer.set_metadata("payload.leakage.channels", "HW(m),HW(y)");
        buffer.set_payload(payload);

        const auto file = (root / "poi.h5").string();
        extras_plugin::BufferFileSink sink(codecs);
        sink.set_property("path", file);
        (void)sink.process(buffer);

        extras_plugin::BufferFileSrc src(codecs);
        src.set_property("path", file);
        const auto loaded = src.process(std::nullopt);
        const auto loaded_payload = loaded->payload_as<crypto_plugin::CorrelationPoiPayload>();
        if (!expect(loaded_payload != nullptr && loaded_payload->result_count() == 2,
                "PoI payload result count did not round-trip")) {
            return 1;
        }
        if (!expect(loaded_payload->result(0).unit == 3
                    && loaded_payload->result(1).unit == 5,
                "PoI byte indexes did not round-trip")) {
            fail();
        }
        if (!expect(loaded_payload->score_name() == "correlation", "PoI score name did not round-trip")) {
            fail();
        }
        if (!expect(torch::equal(loaded_payload->result(0).result, results[0].result),
                "PoI result tensor did not round-trip")) {
            fail();
        }
        if (!expect(loaded->metadata("payload.poi.method") == crypto_plugin::pearson_poi_method_id,
                "PoI method metadata did not round-trip")) {
            fail();
        }
    }

    // --- error paths ---
    {
        auto empty_codecs = std::make_shared<leakflow::PayloadCodecRegistry>();
        extras_plugin::BufferFileSink sink(empty_codecs);
        sink.set_property("path", (root / "unknown.h5").string());
        auto payload = std::make_shared<leakflow::base::TorchTensorPayload>(torch::zeros({2}));
        leakflow::Buffer buffer(payload->caps());
        buffer.set_payload(payload);
        if (!expect(throws([&] { (void)sink.process(buffer); }),
                "BufferFileSink accepted a payload with no registered codec")) {
            fail();
        }

        extras_plugin::BufferFileSink no_path(codecs);
        auto payload2 = std::make_shared<leakflow::base::TorchTensorPayload>(torch::zeros({2}));
        leakflow::Buffer buffer2(payload2->caps());
        buffer2.set_payload(payload2);
        if (!expect(throws([&] { (void)no_path.process(buffer2); }),
                "BufferFileSink accepted an empty path")) {
            fail();
        }
    }

    std::filesystem::remove_all(root);
    return status;
}
