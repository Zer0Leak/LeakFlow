#include "leakflow/plugins/ml/gaussian_mixture_element.hpp"

#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/buffer.hpp"
#include "leakflow/core/progress_sink.hpp"

#include <cstdint>
#include <iostream>
#include <stop_token>
#include <torch/torch.h>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

leakflow::Buffer torch_buffer(torch::Tensor tensor)
{
    auto payload = leakflow::base::TorchTensorPayload(std::move(tensor));
    leakflow::Buffer buffer{payload.caps()};
    buffer.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    return buffer;
}

class CapturingProgressSink final : public leakflow::ProgressSink {
public:
    void report(leakflow::Element&, const leakflow::ElementProgress& progress) override
    {
        reports.push_back(progress);
    }

    std::vector<leakflow::ElementProgress> reports;
};

} // namespace

int main()
{
    torch::manual_seed(20);
    constexpr std::int64_t k = 3;
    constexpr std::int64_t n = 4;
    constexpr std::int64_t per = 60;
    std::vector<torch::Tensor> parts;
    for (std::int64_t c = 0; c < k; ++c) {
        const auto centre = torch::full({n}, static_cast<double>(c) * 20.0, torch::kFloat64);
        parts.push_back(centre.to(torch::kFloat32) + torch::randn({per, n}));
    }
    const auto features = torch::cat(parts, 0); // [180, 4], well separated

    leakflow::plugins::ml::GaussianMixtureElement element;
    element.set_property("n_components", std::int64_t{k});
    element.set_property("covariance_type", std::string("diagonal"));
    element.set_property("n_init", std::int64_t{3});
    element.set_property("seed", std::int64_t{7});
    CapturingProgressSink progress;
    element.set_progress_sink(&progress);

    const auto output = element.process(torch_buffer(features));
    if (!expect(output.has_value(), "element produced no output")) {
        return 1;
    }
    const auto payload = output->payload_as<leakflow::base::TorchTensorPayload>();
    if (!expect(payload != nullptr, "output is not a TorchTensorPayload")) {
        return 1;
    }
    const auto labels = payload->tensor();
    if (!expect(labels.scalar_type() == torch::kLong, "labels are not int64")) {
        return 1;
    }
    if (!expect(labels.sizes() == torch::IntArrayRef({k * per}), "labels shape wrong")) {
        return 1;
    }

    // Well-separated data: each true block gets one label, and the three are distinct.
    const auto cpu = labels.to(torch::kCPU).contiguous();
    const auto acc = cpu.accessor<std::int64_t, 1>();
    std::vector<std::int64_t> block_label(k);
    for (std::int64_t c = 0; c < k; ++c) {
        const auto first = acc[c * per];
        block_label[static_cast<std::size_t>(c)] = first;
        for (std::int64_t i = 0; i < per; ++i) {
            if (!expect(acc[c * per + i] == first, "a true cluster was split across labels")) {
                return 1;
            }
        }
    }
    if (!expect(block_label[0] != block_label[1] && block_label[1] != block_label[2]
                    && block_label[0] != block_label[2],
                "true clusters were merged into one label")) {
        return 1;
    }

    if (!expect(output->metadata_or("payload.cluster.method", "") == "gaussian-mixture", "method metadata wrong")) {
        return 1;
    }
    if (!expect(output->metadata_or("payload.cluster.n_components", "") == "3", "n_components metadata wrong")) {
        return 1;
    }
    if (!expect(output->metadata_or("payload.cluster.converged", "") == "true", "converged metadata wrong")) {
        return 1;
    }
    if (!expect(output->metadata_or("payload.layout", "") == "observation",
            "labels payload layout wrong")) {
        return 1;
    }
    if (!expect(progress.reports.size() >= 2, "fit did not report progress start and completion")) {
        return 1;
    }
    if (!expect(progress.reports.front().fraction == 0.0 && progress.reports.front().message == "starting",
            "first fit progress report was not the initial 0% state")) {
        return 1;
    }
    if (!expect(progress.reports.back().fraction == 1.0 && progress.reports.back().message == "done"
            && progress.reports.back().status == leakflow::ProgressStatus::Completed,
            "last fit progress report was not completion")) {
        return 1;
    }

    leakflow::plugins::ml::GaussianMixtureElement cancelled_element;
    cancelled_element.set_property("n_components", std::int64_t{k});
    cancelled_element.set_property("covariance_type", std::string("diagonal"));
    CapturingProgressSink cancelled_progress;
    cancelled_element.set_progress_sink(&cancelled_progress);
    std::stop_source stop;
    stop.request_stop();
    cancelled_element.set_stop_token(stop.get_token());

    const auto cancelled_output = cancelled_element.process(torch_buffer(features));
    if (!expect(!cancelled_output.has_value(), "pre-stopped fit emitted an output")) {
        return 1;
    }
    if (!expect(cancelled_progress.reports.size() == 1,
            "pre-stopped fit should report only terminal cancellation")) {
        return 1;
    }
    if (!expect(cancelled_progress.reports.back().fraction == 1.0
            && cancelled_progress.reports.back().message == "cancelled"
            && cancelled_progress.reports.back().status == leakflow::ProgressStatus::Cancelled,
            "pre-stopped fit did not report terminal cancellation")) {
        return 1;
    }

    std::cout << "gaussian_mixture_element tests passed\n";
    return 0;
}
