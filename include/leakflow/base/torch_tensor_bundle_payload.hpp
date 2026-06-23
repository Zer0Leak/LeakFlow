#pragma once

#include "leakflow/core/payload.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <vector>

namespace leakflow::base {

class TorchTensorBundlePayload final : public Payload {
public:
    using TensorMap = std::map<std::string, std::shared_ptr<TorchTensorPayload>, std::less<>>;

    [[nodiscard]] std::string type_name() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    void set(std::string name, std::shared_ptr<TorchTensorPayload> payload);
    void set(std::string name, torch::Tensor tensor);

    [[nodiscard]] bool has(std::string_view name) const;
    [[nodiscard]] std::shared_ptr<TorchTensorPayload> payload(std::string_view name);
    [[nodiscard]] std::shared_ptr<const TorchTensorPayload> payload(std::string_view name) const;

    [[nodiscard]] torch::Tensor& tensor(std::string_view name);
    [[nodiscard]] const torch::Tensor& tensor(std::string_view name) const;

    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] const TensorMap& payloads() const;

private:
    TensorMap tensors_;
};

} // namespace leakflow::base
