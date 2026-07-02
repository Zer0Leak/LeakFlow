#pragma once

#include "leakflow/core/caps.hpp"
#include "leakflow/core/payload.hpp"

#include <cstdint>
#include <string>
#include <torch/torch.h>

namespace leakflow::base {

class TorchTensorPayload final : public Payload {
public:
    explicit TorchTensorPayload(torch::Tensor tensor);

    [[nodiscard]] std::string type_name() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    [[nodiscard]] const torch::Tensor& tensor() const;
    [[nodiscard]] torch::Tensor& tensor();

    [[nodiscard]] torch::Device device() const;
    [[nodiscard]] torch::ScalarType dtype() const;
    [[nodiscard]] std::string dtype_name() const;
    [[nodiscard]] std::string device_name() const;
    [[nodiscard]] c10::IntArrayRef shape() const;
    [[nodiscard]] c10::IntArrayRef strides() const;
    [[nodiscard]] std::int64_t rank() const;
    [[nodiscard]] std::int64_t element_count() const;
    [[nodiscard]] bool is_cpu() const;
    [[nodiscard]] bool is_cuda() const;
    [[nodiscard]] bool is_contiguous() const;
    [[nodiscard]] Caps caps() const;

private:
    torch::Tensor tensor_;
};

[[nodiscard]] std::string torch_dtype_name(torch::ScalarType dtype);

} // namespace leakflow::base
