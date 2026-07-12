#include "leakflow/plugins/ml/gaussian_mixture_element.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/ml/gaussian_mixture.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>

namespace leakflow::plugins::ml {
namespace {

[[nodiscard]] std::int64_t int_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
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

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] leakflow::ml::GaussianMixtureOptions options_from(const Element& element)
{
    leakflow::ml::GaussianMixtureOptions options;
    options.n_components = int_property_or(element, "n_components", 8);
    options.covariance_type = leakflow::ml::gmm_covariance_type_for(
        string_property_or(element, "covariance_type", "full"));
    options.init_method = leakflow::ml::gmm_init_method_for(string_property_or(element, "init", "kmeans++"));
    options.n_init = int_property_or(element, "n_init", 1);
    options.max_iter = int_property_or(element, "max_iter", 100);
    options.reg_covar = double_property_or(element, "reg_covar", 1.0e-6);
    const auto seed = int_property_or(element, "seed", 0);
    if (seed >= 0) {
        options.seed = static_cast<std::uint64_t>(seed);
    }
    return options;
}

} // namespace

ElementDescriptor GaussianMixtureElement::descriptor()
{
    return {
        .type_name = "GaussianMixture",
        .klass = "Analyze/Clustering/GMM",
        .purpose = "fit a Gaussian mixture per unit over a whole-dataset feature tensor and emit cluster labels",
        .input_pads = {
            Pad("features", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .output_pads = {
            Pad("labels", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .property_specs = {
            PropertySpec("n_components", std::int64_t{8}, "number of mixture components (clusters) per unit",
                "", IntRangeConstraint{1, 1000000}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"labels"},
                }),
            PropertySpec("covariance_type", std::string("full"), "component covariance form",
                "", StringEnumConstraint{{"full", "diagonal"}}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"labels"},
                }),
            PropertySpec("init", std::string("kmeans++"), "initialisation method",
                "", StringEnumConstraint{{"kmeans++", "random_from_data", "random"}}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"labels"},
                }),
            PropertySpec("n_init", std::int64_t{1}, "number of EM restarts (best kept per unit)",
                "", IntRangeConstraint{1, 100}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"labels"},
                }),
            PropertySpec("max_iter", std::int64_t{100}, "maximum EM iterations (0 = init only)",
                "", IntRangeConstraint{0, 100000}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"labels"},
                }),
            PropertySpec("reg_covar", 1.0e-6, "covariance diagonal regularisation",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"labels"},
                }),
            PropertySpec("seed", std::int64_t{0},
                "init seed; deterministic (>=0) by default so replay reproduces (-1 = nondeterministic)",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"labels"},
                }),
            // UX-only: how often the fit pushes a progress report to the --graph bar. Does not
            // affect the labels, so it carries no invalidation effect.
            PropertySpec("progress_every", std::int64_t{10},
                "report fit progress every N EM iterations (0 disables iteration reports)",
                "", IntRangeConstraint{0, 100000}, "", PropertyEffect{}),
        },
        .keywords = {"gmm", "gaussian", "mixture", "cluster", "em", "ml"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.cluster.method", std::string(), "clustering method", {"gaussian-mixture"}),
            make_element_metadata_descriptor(
                "payload.cluster.n_components", std::int64_t{}, "number of clusters", {"81"}),
            make_element_metadata_descriptor(
                "payload.cluster.covariance_type", std::string(), "component covariance form", {"full", "diagonal"}),
            make_element_metadata_descriptor(
                "payload.cluster.converged", std::string(), "whether every unit converged", {"true", "false"}),
        },
    };
}

GaussianMixtureElement::GaussianMixtureElement(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> GaussianMixtureElement::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("GaussianMixture requires an input buffer");
    }
    if (input->caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument("GaussianMixture features input must have leakflow/torch-tensor caps");
    }
    const auto payload = input->payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument("GaussianMixture features input must carry a TorchTensorPayload");
    }

    const auto options = options_from(*this);
    leakflow::ml::GaussianMixture model(options);
    bool cancelled = false;

    const auto report_cancelled = [this, &options]() {
        report_progress(1.0, "cancelled", 0, static_cast<std::uint64_t>(options.max_iter),
            leakflow::ProgressStatus::Cancelled);
    };

    if (!cooperative_checkpoint()) {
        report_cancelled();
        return std::nullopt;
    }

    // Bridge the ml-lib progress callback to the framework progress bar. We gate on
    // progress_every here (restart boundaries, stage changes, and the final iteration always
    // pass); report_progress further coalesces to ~30 Hz. Returning false cancels the fit, which
    // we wire to the element's stop token so Ctrl+C / window-close interrupts a long fit.
    const auto progress_every = int_property_or(*this, "progress_every", 10);
    const auto on_checkpoint = [this, &cancelled]() {
        if (!cooperative_checkpoint()) {
            cancelled = true;
            return false;
        }
        return true;
    };
    const auto on_progress = [this, progress_every, &on_checkpoint](const leakflow::ml::GmmProgress& p) -> bool {
        if (!on_checkpoint()) {
            return false;
        }
        const auto report = progress_every > 0
            && (p.iter <= 1 || p.iter == p.max_iter || (p.iter % progress_every) == 0);
        if (report) {
            std::ostringstream message;
            if (p.stage == "constrained") {
                message << "sinkhorn " << p.iter << "/" << p.max_iter;
            } else {
                if (p.restarts > 1) {
                    message << "restart " << (p.restart + 1) << "/" << p.restarts << " - ";
                }
                message << "iter " << p.iter << "/" << p.max_iter;
            }
            report_progress(p.fraction, message.str(), static_cast<std::uint64_t>(p.iter),
                static_cast<std::uint64_t>(p.max_iter));
        }
        return true;
    };

    // Initialisation happens before the ML callback's first EM iteration and can take long
    // enough to be visible. Publish 0% immediately so observers can show the progress bar while
    // it runs; report_progress keeps its normal throttling for the callbacks that follow.
    report_progress(0.0, "starting", 0, static_cast<std::uint64_t>(options.max_iter));
    const auto fit = model.fit(payload->tensor(), on_progress, on_checkpoint);
    if (cancelled || !cooperative_checkpoint()) {
        report_cancelled();
        return std::nullopt;
    }
    report_progress(1.0, "done", 0, 0, leakflow::ProgressStatus::Completed);
    const auto labels = fit.labels.to(torch::kInt64).contiguous();
    const bool converged = fit.converged.all().item<bool>();

    auto label_payload = leakflow::base::TorchTensorPayload(labels);
    Buffer output{label_payload.caps()};
    forward_metadata(*input, profile_for_klass(element_kclass()), output, "features", name());
    output.set_metadata("payload.cluster.method", "gaussian-mixture");
    output.set_metadata("payload.cluster.n_components", std::to_string(options.n_components));
    output.set_metadata("payload.cluster.covariance_type",
        options.covariance_type == leakflow::ml::GmmCovarianceType::Full ? "full" : "diagonal");
    output.set_metadata("payload.cluster.converged", converged ? "true" : "false");
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(label_payload)));

    auto record = make_log_record(log::LogLevel::Debug, "element", "fit gaussian mixture");
    record.fields.emplace("clusters", std::to_string(options.n_components));
    record.fields.emplace("converged", converged ? "true" : "false");
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::ml
