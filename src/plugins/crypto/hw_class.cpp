#include "leakflow/plugins/crypto/hw_class.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>

namespace leakflow::plugins::crypto {
namespace {

[[nodiscard]] std::int64_t int_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }
    return fallback;
}

} // namespace

ElementDescriptor HwClass::descriptor()
{
    return {
        .type_name = "HwClass",
        .klass = "Convert/SCA/HwClass",
        .purpose = "combine HW(m) and HW(y) leakage channels into a single true class label per trace",
        .input_pads = {
            Pad("leakage", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .output_pads = {
            Pad("class", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .property_specs = {
            PropertySpec("hw_max", std::int64_t{8}, "maximum Hamming weight per variable (radix = hw_max + 1)",
                "", IntRangeConstraint{1, 4096}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"class"},
                }),
        },
        .keywords = {"hamming", "weight", "class", "truth", "sca", "aes"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.class.count", std::int64_t{}, "number of distinct classes", {"81"}),
        },
    };
}

HwClass::HwClass(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> HwClass::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("HwClass requires an input buffer");
    }
    if (input->caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument("HwClass leakage input must have leakflow/torch-tensor caps");
    }
    const auto payload = input->payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument("HwClass leakage input must carry a TorchTensorPayload");
    }
    const auto leakage = payload->tensor();
    if (leakage.dim() != 3 || leakage.size(2) < 2) {
        throw std::invalid_argument("HwClass expects a [B, N, C>=2] leakage tensor (HW(m), HW(y))");
    }

    const auto hw_max = int_property_or(*this, "hw_max", 8);
    const auto radix = hw_max + 1;
    const auto hw_m = leakage.select(2, 0).to(torch::kLong);
    const auto hw_y = leakage.select(2, 1).to(torch::kLong);
    const auto classes = (radix * hw_m + hw_y).contiguous(); // [B, N]

    auto class_payload = leakflow::base::TorchTensorPayload(classes);
    Buffer output{class_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "leakage", name());
    output.set_metadata("payload.class.count", std::to_string(radix * radix));
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(class_payload)));
    return output;
}

} // namespace leakflow::plugins::crypto
