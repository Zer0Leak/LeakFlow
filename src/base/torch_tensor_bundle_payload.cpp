#include "leakflow/base/torch_tensor_bundle_payload.hpp"

#include "leakflow/log/logger.hpp"

#include <c10/core/ScalarType.h>

#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace leakflow::base {
namespace {

void validate_name(std::string_view name)
{
    if (name.empty()) {
        throw std::invalid_argument("TorchTensorBundlePayload tensor name cannot be empty");
    }
}

} // namespace

std::string TorchTensorBundlePayload::type_name() const
{
    return "leakflow/torch-tensor-bundle";
}

std::string TorchTensorBundlePayload::layout() const
{
    if (tensors_.empty()) {
        return "empty";
    }

    std::ostringstream output;
    auto first = true;
    for (const auto& [name, payload] : tensors_) {
        if (!first) {
            output << ';';
        }
        first = false;
        output << name << '=' << payload->layout();
    }
    return output.str();
}

void TorchTensorBundlePayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("tensors", summary_size(tensors_.size()), SummaryValueRole::Number);

    if (summary_level >= 1) {
        section.add_field("names", summary_list(names()), SummaryValueRole::Text);
    }

    if (summary_level >= 2) {
        for (const auto& [name, payload] : tensors_) {
            auto& tensor = section.add_field(name, payload->type_name(), SummaryValueRole::TypeName);
            tensor.add_child("rank", summary_integer(payload->rank()), SummaryValueRole::Number);
            tensor.add_child(
                "shape",
                summary_list_from_int_array(payload->shape().data(), payload->shape().size()),
                SummaryValueRole::Number);
            tensor.add_child("dtype", c10::toString(payload->dtype()), SummaryValueRole::TypeName);
        }
    }
}

void TorchTensorBundlePayload::set(std::string name, std::shared_ptr<TorchTensorPayload> payload)
{
    validate_name(name);
    if (!payload) {
        throw std::invalid_argument("TorchTensorBundlePayload tensor payload cannot be null");
    }

    tensors_[std::move(name)] = std::move(payload);

    log::LogRecord record{
        .level = log::LogLevel::Trace,
        .component = "base",
        .message = "stored tensor payload in bundle",
        .fields = {
            {"tensors", std::to_string(tensors_.size())},
        },
    };
    log::write(std::move(record));
}

void TorchTensorBundlePayload::set(std::string name, torch::Tensor tensor)
{
    set(std::move(name), std::make_shared<TorchTensorPayload>(std::move(tensor)));
}

bool TorchTensorBundlePayload::has(std::string_view name) const
{
    return tensors_.contains(name);
}

std::shared_ptr<TorchTensorPayload> TorchTensorBundlePayload::payload(std::string_view name)
{
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::out_of_range("unknown tensor payload name");
    }

    return found->second;
}

std::shared_ptr<const TorchTensorPayload> TorchTensorBundlePayload::payload(std::string_view name) const
{
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::out_of_range("unknown tensor payload name");
    }

    return found->second;
}

torch::Tensor& TorchTensorBundlePayload::tensor(std::string_view name)
{
    return payload(name)->tensor();
}

const torch::Tensor& TorchTensorBundlePayload::tensor(std::string_view name) const
{
    return payload(name)->tensor();
}

std::vector<std::string> TorchTensorBundlePayload::names() const
{
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& [name, payload] : tensors_) {
        result.push_back(name);
    }
    return result;
}

std::size_t TorchTensorBundlePayload::size() const
{
    return tensors_.size();
}

const TorchTensorBundlePayload::TensorMap& TorchTensorBundlePayload::payloads() const
{
    return tensors_;
}

} // namespace leakflow::base
