#include "leakflow/plugins/ml/clustering_stats.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/core/metadata_forwarding.hpp"
#include "leakflow/ml/clustering_metrics.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>

namespace leakflow::plugins::ml {
namespace {

[[nodiscard]] const Buffer& required_input(
    const ElementInputs& inputs, std::string_view pad, std::string_view element)
{
    const auto found = inputs.find(std::string(pad));
    if (found == inputs.end() || !found->second) {
        throw std::invalid_argument(std::string(element) + " requires connected input pad " + std::string(pad));
    }
    return *found->second;
}

[[nodiscard]] torch::Tensor int_tensor_input(const Buffer& buffer, std::string_view pad)
{
    if (buffer.caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument(std::string(pad) + " input must have leakflow/torch-tensor caps");
    }
    const auto payload = buffer.payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument(std::string(pad) + " input must carry a TorchTensorPayload");
    }
    return payload->tensor().to(torch::kLong).contiguous();
}

[[nodiscard]] std::int64_t int_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] std::string join_units(const std::vector<std::int64_t>& units)
{
    std::string text;
    for (std::size_t index = 0; index < units.size(); ++index) {
        if (index != 0) {
            text += ", ";
        }
        text += std::to_string(units[index]);
    }
    return text;
}

[[nodiscard]] std::string format(double value)
{
    return std::to_string(value);
}

// A [U] (or scalar) tensor formatted as a comma-separated per-unit list.
[[nodiscard]] std::string comma_list(const torch::Tensor& values)
{
    const auto flat = (values.dim() == 0 ? values.unsqueeze(0) : values).to(torch::kFloat64).to(torch::kCPU).contiguous();
    const auto accessor = flat.accessor<double, 1>();
    std::string out;
    for (std::int64_t i = 0; i < flat.size(0); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(accessor[i]);
    }
    return out;
}

} // namespace

ElementDescriptor ClusteringStats::descriptor()
{
    return {
        .type_name = "ClusteringStats",
        .klass = "Analyze/Evaluation/Clustering",
        .purpose = "score a clustering against true classes; emit the reordered confusion + ARI/NMI/purity/accuracy",
        .input_pads = {
            Pad("labels", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
            Pad("truth", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .output_pads = {
            Pad("stats", PadDirection::Output, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .property_specs = {
            PropertySpec("n_classes", std::int64_t{-1}, "number of true classes (-1 = infer from truth)",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"stats"},
                }),
            PropertySpec("n_clusters", std::int64_t{-1}, "number of clusters (-1 = infer from labels)",
                "", std::monostate{}, "",
                PropertyEffect{
                    .kind = PropertyEffectKind::PayloadOutput,
                    .scope = PropertyInvalidationScope::Downstream,
                    .output_pads = {"stats"},
                }),
        },
        .keywords = {"clustering", "evaluation", "confusion", "ari", "nmi", "accuracy", "ml"},
        .metadata_set_by_element = {
            make_element_metadata_descriptor(
                "payload.layout", std::string(), "logical axes of the reordered confusion tensor",
                {"true_class/cluster", "unit/true_class/cluster"}),
            make_element_metadata_descriptor(
                "payload.cluster_stats.accuracy", std::string(), "mean matched accuracy over units", {"0.68"}),
            make_element_metadata_descriptor(
                "payload.cluster_stats.ari", std::string(), "mean adjusted Rand index", {"0.62"}),
            make_element_metadata_descriptor(
                "payload.cluster_stats.nmi", std::string(), "mean normalized mutual information", {"0.80"}),
            make_element_metadata_descriptor(
                "payload.cluster_stats.purity", std::string(), "mean cluster purity", {"0.89"}),
        },
    };
}

ClusteringStats::ClusteringStats(std::string name)
    : Element(std::move(name))
{
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> ClusteringStats::process(std::optional<Buffer>)
{
    throw std::invalid_argument("ClusteringStats requires named labels and truth inputs");
}

std::optional<Buffer> ClusteringStats::process_inputs(ElementInputs inputs)
{
    const auto& labels_buffer = required_input(inputs, "labels", "ClusteringStats");
    const auto& truth_buffer = required_input(inputs, "truth", "ClusteringStats");
    auto labels = int_tensor_input(labels_buffer, "labels");
    auto truth = int_tensor_input(truth_buffer, "truth");

    // Unit-axis alignment before the per-unit comparison. Shape equality is
    // necessary but not sufficient: two size-1 unit axes can still be different
    // units (cluster labels for byte 1 vs Hamming-weight truth for byte 0), which
    // scores silently wrong. When both inputs label their leading (unit) axis,
    // align by unit id -- error on disjoint units, warn and score the shared units
    // on a partial overlap. Unlabeled inputs fall back to the plain shape check.
    std::vector<std::int64_t> output_units;
    const auto labels_units = labels_buffer.units().to_vector();
    const auto truth_units = truth_buffer.units().to_vector();
    const bool have_units = !labels_units.empty() && !truth_units.empty()
        && labels_units.size() == static_cast<std::size_t>(labels.size(0))
        && truth_units.size() == static_cast<std::size_t>(truth.size(0));
    if (have_units) {
        const auto alignment = align_labels(labels_units, truth_units);
        if (alignment.shared.empty()) {
            throw std::invalid_argument(identity_for_error() + ": labels and truth share no units (labels ["
                + join_units(labels_units) + "], truth [" + join_units(truth_units)
                + "]); they describe different units, so the comparison is undefined");
        }
        if (!alignment.identical) {
            auto warning = make_log_record(log::LogLevel::Warning, "element",
                "labels and truth cover different units; scoring the "
                    + std::to_string(alignment.shared.size()) + " shared unit(s)");
            warning.fields.emplace("labels_units", join_units(labels_units));
            warning.fields.emplace("truth_units", join_units(truth_units));
            warning.fields.emplace("shared_units", join_units(alignment.shared));
            leakflow::log::write(std::move(warning));
            const auto options = torch::TensorOptions().dtype(torch::kInt64);
            labels = labels.index_select(0, torch::tensor(alignment.a_indices, options)).contiguous();
            truth = truth.index_select(0, torch::tensor(alignment.b_indices, options)).contiguous();
        }
        output_units = alignment.shared;
    }

    if (labels.sizes() != truth.sizes()) {
        throw std::invalid_argument(identity_for_error() + ": labels and truth must have the same shape");
    }

    const auto n_classes_prop = int_property_or(*this, "n_classes", -1);
    const auto n_clusters_prop = int_property_or(*this, "n_clusters", -1);
    const auto n_classes = n_classes_prop > 0 ? n_classes_prop : truth.max().item<std::int64_t>() + 1;
    const auto n_clusters = n_clusters_prop > 0 ? n_clusters_prop : labels.max().item<std::int64_t>() + 1;
    const auto size = std::max(n_classes, n_clusters); // square for Hungarian matching

    const auto confusion = leakflow::ml::confusion_matrix(labels, truth, size, size);
    const auto purity_per_unit = leakflow::ml::cluster_purity(confusion).to(torch::kFloat64);
    const auto ari_per_unit = leakflow::ml::adjusted_rand_index(confusion).to(torch::kFloat64);
    const auto nmi_per_unit = leakflow::ml::normalized_mutual_info(confusion).to(torch::kFloat64);
    const auto matched = leakflow::ml::matched_clustering_scores(confusion);
    const auto accuracy_per_unit = matched.accuracy.to(torch::kFloat64);
    const auto reordered = leakflow::ml::reorder_confusion_columns(confusion, matched.matching).to(torch::kFloat64);

    auto payload = leakflow::base::TorchTensorPayload(reordered.contiguous());
    Buffer output{payload.caps()};
    forward_metadata(inputs, profile_for_klass(element_kclass()), output, name());
    // Means (headline) and per-unit lists (for a HeatmapPlot unit slider / per-byte inspection).
    output.set_metadata("payload.cluster_stats.accuracy", format(accuracy_per_unit.mean().item<double>()));
    output.set_metadata("payload.cluster_stats.ari", format(ari_per_unit.mean().item<double>()));
    output.set_metadata("payload.cluster_stats.nmi", format(nmi_per_unit.mean().item<double>()));
    output.set_metadata("payload.cluster_stats.purity", format(purity_per_unit.mean().item<double>()));
    output.set_metadata("payload.cluster_stats.accuracy_per_unit", comma_list(accuracy_per_unit));
    output.set_metadata("payload.cluster_stats.ari_per_unit", comma_list(ari_per_unit));
    output.set_metadata("payload.cluster_stats.nmi_per_unit", comma_list(nmi_per_unit));
    output.set_metadata("payload.cluster_stats.purity_per_unit", comma_list(purity_per_unit));
    output.set_metadata("payload.cluster_stats.n_classes", std::to_string(n_classes));
    output.set_metadata("payload.cluster_stats.n_clusters", std::to_string(n_clusters));
    output.set_payload(std::make_shared<leakflow::base::TorchTensorPayload>(std::move(payload)));
    output.set_metadata("payload.layout",
        reordered.dim() == 2 ? "true_class/cluster" : "unit/true_class/cluster");
    if (!output_units.empty()) {
        output.set_units(Units::of(std::move(output_units)));
    }

    auto record = make_log_record(log::LogLevel::Debug, "element", "scored clustering vs truth");
    record.fields.emplace("accuracy", format(accuracy_per_unit.mean().item<double>()));
    record.fields.emplace("ari", format(ari_per_unit.mean().item<double>()));
    leakflow::log::write(std::move(record));

    return output;
}

} // namespace leakflow::plugins::ml
