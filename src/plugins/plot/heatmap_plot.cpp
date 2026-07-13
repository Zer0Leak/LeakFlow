#include "leakflow/plugins/plot/heatmap_plot.hpp"

#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/base/torch_tensor_payload.hpp"
#include "leakflow/plot/heatmap_view.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::plot {
namespace {

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] std::int64_t int_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] std::vector<std::string> split_comma(std::string_view text)
{
    std::vector<std::string> values;
    if (text.empty()) {
        return values;
    }
    std::size_t begin = 0;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        if (index == text.size() || text[index] == ',') {
            auto token = text.substr(begin, index - begin);
            const auto first = token.find_first_not_of(" \t");
            const auto last = token.find_last_not_of(" \t");
            values.emplace_back(first == std::string_view::npos ? "" : token.substr(first, last - first + 1));
            begin = index + 1;
        }
    }
    return values;
}

} // namespace

ElementDescriptor HeatmapPlot::descriptor()
{
    return {
        .type_name = "HeatmapPlot",
        .klass = "Sink/Plot/Heatmap",
        .purpose = "render a matrix tensor as an ImPlot heatmap (e.g. a ClusteringStats confusion matrix)",
        .input_pads = {
            Pad("matrix", PadDirection::Input, Caps(leakflow::base::torch_tensor_caps_type)),
        },
        .property_specs = {
            PropertySpec("normalize", std::string("row"), "colour scale: none (raw), row, or column normalised",
                "", StringEnumConstraint{{"none", "row", "col"}}, "", PropertyEffect{}),
            PropertySpec("unit", std::int64_t{-1}, "batched [U,R,C]: -1 = keep all units (slider), >=0 = pin one",
                "", std::monostate{}, "", PropertyEffect{}),
            PropertySpec("title", std::string("Heatmap"), "window title", "", std::monostate{}, "", PropertyEffect{}),
            PropertySpec("row_label", std::string("true class"), "y-axis label", "", std::monostate{}, "",
                PropertyEffect{}),
            PropertySpec("col_label", std::string("cluster"), "x-axis label", "", std::monostate{}, "",
                PropertyEffect{}),
            PropertySpec("caption_from", std::string(""),
                "metadata key of a per-unit comma list to caption each unit (e.g. cluster_stats per-unit accuracy)",
                "", std::monostate{}, "", PropertyEffect{}),
        },
        .keywords = {"heatmap", "plot", "confusion", "matrix", "imgui", "implot"},
    };
}

HeatmapPlot::HeatmapPlot(std::string name)
    : HeatmapPlot(std::make_shared<leakflow::plot::HeatmapView>(), std::move(name))
{
}

HeatmapPlot::HeatmapPlot(std::shared_ptr<leakflow::plot::HeatmapView> view, std::string name)
    : Element(std::move(name)), view_(std::move(view))
{
    if (!view_) {
        throw std::invalid_argument("HeatmapPlot requires a HeatmapView");
    }
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> HeatmapPlot::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("HeatmapPlot requires an input buffer");
    }
    if (input->caps().type() != leakflow::base::torch_tensor_caps_type) {
        throw std::invalid_argument("HeatmapPlot matrix input must have leakflow/torch-tensor caps");
    }
    const auto payload = input->payload_as<leakflow::base::TorchTensorPayload>();
    if (!payload) {
        throw std::invalid_argument("HeatmapPlot matrix input must carry a TorchTensorPayload");
    }

    auto matrix = payload->tensor().to(torch::kFloat64);
    if (matrix.dim() == 2) {
        matrix = matrix.unsqueeze(0); // [1, R, C]
    }
    if (matrix.dim() != 3) {
        throw std::invalid_argument("HeatmapPlot expects a [R, C] or [U, R, C] matrix");
    }
    const auto unit = int_property_or(*this, "unit", -1);
    if (unit >= 0 && unit < matrix.size(0)) {
        matrix = matrix[unit].unsqueeze(0); // pin one unit -> [1, R, C]
    }

    // Normalise per unit (over the last two axes) for the colour scale.
    const auto normalize = string_property_or(*this, "normalize", "row");
    std::string value_label;
    double vmin = 0.0;
    double vmax = 1.0;
    if (normalize == "row") {
        matrix = matrix / matrix.sum(2, /*keepdim=*/true).clamp_min(1.0e-12);
        value_label = "row-normalised";
    } else if (normalize == "col") {
        matrix = matrix / matrix.sum(1, /*keepdim=*/true).clamp_min(1.0e-12);
        value_label = "column-normalised";
    } else {
        value_label = "value";
        vmin = matrix.min().item<double>();
        vmax = matrix.max().item<double>(); // shared scale across units
    }
    if (vmax <= vmin) {
        vmax = vmin + 1.0;
    }

    matrix = matrix.contiguous();
    const auto units = matrix.size(0);
    const auto rows = matrix.size(1);
    const auto cols = matrix.size(2);
    const auto* ptr = matrix.data_ptr<double>();
    std::vector<double> data(ptr, ptr + units * rows * cols);

    // Optional per-unit captions from a comma-separated metadata list.
    std::vector<std::string> captions;
    const auto caption_key = string_property_or(*this, "caption_from", "");
    if (!caption_key.empty()) {
        const auto values = split_comma(input->metadata_or(caption_key, ""));
        if (static_cast<std::int64_t>(values.size()) == units) {
            captions.reserve(values.size());
            for (std::int64_t u = 0; u < units; ++u) {
                captions.push_back("unit " + std::to_string(u) + ":  " + values[static_cast<std::size_t>(u)]);
            }
        }
    }

    view_->set_matrix(name(), string_property_or(*this, "title", "Heatmap"),
        string_property_or(*this, "row_label", "true class"),
        string_property_or(*this, "col_label", "cluster"), std::move(value_label), std::move(captions), units, rows,
        cols, std::move(data), vmin, vmax);

    return std::nullopt; // sink
}

std::shared_ptr<leakflow::plot::HeatmapView> HeatmapPlot::heatmap_view() const
{
    return view_;
}

} // namespace leakflow::plugins::plot
