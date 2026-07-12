#include "leakflow/core/buffer.hpp"
#include "leakflow/base/torch_tensor_bundle_payload.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

template <typename Exception, typename Function>
bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }

    return false;
}

} // namespace

int main()
{
    leakflow::base::TorchTensorBundlePayload bundle;
    if (!expect(bundle.layout() == "empty", "empty bundle payload layout mismatch")) {
        return 1;
    }
    auto traces = std::make_shared<leakflow::base::TorchTensorPayload>(
        torch::zeros({4, 8}, torch::TensorOptions().dtype(torch::kFloat32)));

    bundle.set("traces", traces);
    bundle.set("key", torch::arange(16, torch::TensorOptions().dtype(torch::kUInt8)));
    bundle.set("plaintexts", torch::zeros({4, 16}, torch::TensorOptions().dtype(torch::kUInt8)));

    if (!expect(bundle.type_name() == "leakflow/torch-tensor-bundle", "bundle payload type name changed")) {
        return 1;
    }
    if (!expect(bundle.size() == 3, "bundle size mismatch")) {
        return 1;
    }
    if (!expect(bundle.layout() == "key=axis_0;plaintexts=axis_0/axis_1;traces=axis_0/axis_1",
            "bundle payload layout mismatch")) {
        return 1;
    }
    if (!expect(bundle.has("traces"), "bundle missing traces tensor")) {
        return 1;
    }
    if (!expect(!bundle.has("missing"), "bundle reported missing tensor as present")) {
        return 1;
    }
    if (!expect(bundle.payload("traces") == traces, "bundle did not preserve shared payload identity")) {
        return 1;
    }
    if (!expect(bundle.tensor("plaintexts").scalar_type() == torch::kUInt8, "bundle tensor dtype mismatch")) {
        return 1;
    }

    const auto names = bundle.names();
    if (!expect(names == std::vector<std::string>{"key", "plaintexts", "traces"},
            "bundle names are not deterministic")) {
        return 1;
    }

    leakflow::SummarySection summary_section("Payload");
    bundle.describe(summary_section, 2);
    if (!expect(summary_section.fields.size() >= 6, "bundle summary did not include tensor entries")) {
        return 1;
    }
    if (!expect(summary_section.fields[1].label == "tensors", "bundle summary did not include tensor count")) {
        return 1;
    }
    if (!expect(summary_section.fields[2].label == "names", "bundle summary did not include tensor names")) {
        return 1;
    }

    if (!expect(throws_exception<std::invalid_argument>([] {
            leakflow::base::TorchTensorBundlePayload invalid;
            invalid.set("", torch::zeros({1}));
        }),
            "empty bundle tensor names should be rejected")) {
        return 1;
    }
    if (!expect(throws_exception<std::invalid_argument>([] {
            leakflow::base::TorchTensorBundlePayload invalid;
            invalid.set("null", nullptr);
        }),
            "null bundle tensor payloads should be rejected")) {
        return 1;
    }
    if (!expect(throws_exception<std::invalid_argument>([] {
            leakflow::base::TorchTensorBundlePayload invalid;
            invalid.set("undefined", torch::Tensor());
        }),
            "undefined bundle tensor payloads should be rejected")) {
        return 1;
    }
    if (!expect(throws_exception<std::out_of_range>([&bundle] {
            (void)bundle.payload("missing");
        }),
            "missing bundle tensor lookup should throw")) {
        return 1;
    }

    leakflow::Buffer buffer(leakflow::Caps("leakflow/torch-tensor-bundle"));
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorBundlePayload>(bundle));

    const auto roundtrip = buffer.payload_as<leakflow::base::TorchTensorBundlePayload>();
    if (!expect(roundtrip != nullptr, "buffer did not preserve TorchTensorBundlePayload type")) {
        return 1;
    }
    if (!expect(roundtrip->has("key"), "buffer bundle payload lost named tensor")) {
        return 1;
    }
    if (!expect(buffer.metadata("payload.layout") ==
                "key=axis_0;plaintexts=axis_0/axis_1;traces=axis_0/axis_1",
            "buffer did not publish bundle payload layout")) {
        return 1;
    }

    return 0;
}
