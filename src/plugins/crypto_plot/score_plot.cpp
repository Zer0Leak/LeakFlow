#include "leakflow/plugins/crypto_plot/score_plot.hpp"

#include "leakflow/plugins/crypto/attack_payload.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace leakflow::plugins::crypto_plot {
namespace {

namespace crypto = leakflow::plugins::crypto;

[[nodiscard]] std::string string_property_or(const Element& element, std::string_view name, std::string fallback)
{
    if (const auto value = element.property_as<std::string>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] std::int64_t integer_property_or(const Element& element, std::string_view name, std::int64_t fallback)
{
    if (const auto value = element.property_as<std::int64_t>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] bool bool_property_or(const Element& element, std::string_view name, bool fallback)
{
    if (const auto value = element.property_as<bool>(name)) {
        return *value;
    }
    return fallback;
}

[[nodiscard]] std::vector<std::string> string_list_property_or(
    const Element& element, std::string_view name, std::vector<std::string> fallback)
{
    if (const auto value = element.property_as<std::vector<std::string>>(name)) {
        return *value;
    }
    return std::move(fallback);
}

[[nodiscard]] std::shared_ptr<crypto::AttackStatsPayload> stats_payload_for(const Buffer& input)
{
    if (input.caps().type() != crypto::attack_stats_caps_type) {
        throw std::invalid_argument("ScorePlot requires leakflow/attack-stats input caps");
    }
    auto payload = input.payload_as<crypto::AttackStatsPayload>();
    if (!payload) {
        throw std::invalid_argument("ScorePlot requires an AttackStatsPayload");
    }
    return payload;
}

// x for this buffer: the cumulative observation count stamped by CpaAttack, or a
// running step index when that metadata is absent.
[[nodiscard]] double observation_x(const Buffer& input, std::int64_t& step)
{
    if (input.has_metadata("attack.observation_count")) {
        const auto& text = input.metadata("attack.observation_count");
        std::size_t consumed = 0;
        const auto value = std::stoll(text, &consumed);
        if (consumed == text.size() && value > 0) {
            return static_cast<double>(value);
        }
    }
    return static_cast<double>(++step);
}

[[nodiscard]] leakflow::plot::TracePlotColor unit_color(std::int64_t unit)
{
    static constexpr std::array<leakflow::plot::TracePlotColor, 8> palette{{
        {0.05F, 0.80F, 0.35F},
        {0.25F, 0.55F, 1.00F},
        {1.00F, 0.60F, 0.20F},
        {0.95F, 0.35F, 0.55F},
        {0.65F, 0.45F, 1.00F},
        {0.15F, 0.85F, 0.85F},
        {0.95F, 0.85F, 0.25F},
        {0.80F, 0.80F, 0.85F},
    }};
    return palette[static_cast<std::size_t>(std::max<std::int64_t>(unit, 0)) % palette.size()];
}

// A metric name -> the corresponding per-unit CPU float64 tensor. topk_* metrics
// use the top-1 column [U,0]; true_rank is the correct-key rank (truth only).
[[nodiscard]] std::optional<torch::Tensor> metric_tensor(const crypto::AttackStatsPayload& payload,
                                                         std::string_view metric)
{
    const auto to_top1 = [](const torch::Tensor& topk) {
        return topk.to(torch::kCPU).to(torch::kFloat64).contiguous().select(1, 0).contiguous();
    };
    if (metric == "score") {
        return to_top1(payload.topk_score());
    }
    if (metric == "relative_margin") {
        return to_top1(payload.topk_relative_margin());
    }
    if (metric == "margin") {
        return to_top1(payload.topk_margin());
    }
    if (metric == "z_score") {
        return to_top1(payload.topk_z_score());
    }
    if (metric == "robust_z_score") {
        return to_top1(payload.topk_robust_z_score());
    }
    if (metric == "top_k_separation") {
        return to_top1(payload.topk_separation());
    }
    if (metric == "true_rank") {
        if (!payload.has_truth()) {
            return std::nullopt;
        }
        return payload.true_rank()->to(torch::kCPU).to(torch::kFloat64).contiguous();
    }
    return std::nullopt; // unknown metric: skip its panel
}

[[nodiscard]] std::string unit_label(const std::vector<std::int64_t>& unit_indexes, std::int64_t unit)
{
    if (unit >= 0 && static_cast<std::size_t>(unit) < unit_indexes.size()) {
        return std::to_string(unit_indexes[static_cast<std::size_t>(unit)]);
    }
    return std::to_string(unit);
}

[[nodiscard]] std::string format_double(double value)
{
    std::ostringstream output;
    output.precision(4);
    output << value;
    return output.str();
}

std::optional<Buffer> capture_score(Element& element, leakflow::plot::ScoreView& view, const Buffer& input,
                                    std::int64_t& step)
{
    const auto payload = stats_payload_for(input);
    const auto group = string_property_or(element, "group", "cpa");
    const auto title = string_property_or(element, "title", "");
    const auto metrics = string_list_property_or(element, "metrics", {"score", "relative_margin"});
    const auto max_units = std::max<std::int64_t>(integer_property_or(element, "max_units", 64), 1);
    const auto show_second_score = bool_property_or(element, "show_second_score", false);

    const auto x = observation_x(input, step);
    const auto unit_count = std::min(payload->unit_count(), max_units);
    const bool has_truth = payload->has_truth();

    // Pre-convert the referenced metric tensors once per buffer.
    std::vector<std::pair<std::string, torch::Tensor>> panels;
    bool has_score_panel = false;
    for (const auto& metric : metrics) {
        if (auto tensor = metric_tensor(*payload, metric)) {
            has_score_panel = has_score_panel || metric == "score";
            panels.emplace_back(metric, std::move(*tensor));
        }
    }
    // Second-best score (topk_score column 1), for the "score" panel only. Always
    // accumulated when available (top_k >= 2); display is gated by show_second_score.
    const auto topk_score = payload->topk_score().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const bool second_available = has_score_panel && topk_score.size(1) >= 2;
    const auto success = has_truth ? payload->success()->to(torch::kCPU).to(torch::kBool).contiguous() : torch::Tensor{};
    const auto true_rank = has_truth ? payload->true_rank()->to(torch::kCPU).to(torch::kInt64).contiguous()
                                     : torch::Tensor{};
    const auto top1_guess = payload->top1_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto& unit_indexes = payload->unit_indexes();

    std::vector<leakflow::plot::ScorePointUpdate> updates;
    for (std::int64_t unit = 0; unit < unit_count; ++unit) {
        const auto succeeded = has_truth && success[unit].item<bool>();
        const auto marker = !has_truth ? leakflow::plot::TracePlotAnnotationMarker::Circle
                                       : (succeeded ? leakflow::plot::TracePlotAnnotationMarker::Square
                                                    : leakflow::plot::TracePlotAnnotationMarker::Cross);
        const auto label = "byte " + unit_label(unit_indexes, unit);
        const auto color = unit_color(unit);

        // Per-unit metric values, computed once.
        std::vector<std::pair<std::string, double>> metric_values;
        metric_values.reserve(panels.size());
        for (const auto& [metric, tensor] : panels) {
            metric_values.emplace_back(metric, tensor[unit].item<double>());
        }

        // Shared hover fields include EVERY metric, so hovering a point in one panel
        // also shows the other metrics (score shows relative_margin and vice versa).
        std::vector<std::pair<std::string, std::string>> fields;
        fields.emplace_back("byte", unit_label(unit_indexes, unit));
        fields.emplace_back("N", format_double(x));
        fields.emplace_back("guess", std::to_string(top1_guess[unit].item<std::int64_t>()));
        if (has_truth) {
            fields.emplace_back("success", succeeded ? "true" : "false");
            fields.emplace_back("true rank", std::to_string(true_rank[unit].item<std::int64_t>()));
        }
        for (const auto& [metric, value] : metric_values) {
            fields.emplace_back(metric, format_double(value));
        }
        std::optional<double> second_score;
        if (second_available) {
            second_score = topk_score[unit][1].item<double>();
            fields.emplace_back("2nd score", format_double(*second_score));
        }

        for (const auto& [metric, value] : metric_values) {
            updates.push_back(leakflow::plot::ScorePointUpdate{
                .panel = metric,
                .panel_y_label = metric,
                .series = label,
                .color = color,
                .point = leakflow::plot::ScoreSeriesPoint{.x = x, .y = value, .marker = marker, .fields = fields},
            });
        }
        // Secondary "score" series (second-best guess): same label/color as the
        // primary, no success marker. Accumulated regardless; shown per show_secondary.
        if (second_score) {
            updates.push_back(leakflow::plot::ScorePointUpdate{
                .panel = "score",
                .panel_y_label = "score",
                .series = label,
                .color = color,
                .secondary = true,
                .point = leakflow::plot::ScoreSeriesPoint{
                    .x = x, .y = *second_score, .marker = leakflow::plot::TracePlotAnnotationMarker::Circle,
                    .fields = fields},
            });
        }
    }

    view.append_points(element.name(), group, title, "traces (N)", show_second_score, updates);

    auto record = element.make_log_record(log::LogLevel::Debug, "element", "appended score-plot points");
    record.fields.emplace("units", std::to_string(unit_count));
    record.fields.emplace("panels", std::to_string(panels.size()));
    record.fields.emplace("x", format_double(x));
    leakflow::log::write(std::move(record));

    return input;
}

[[nodiscard]] const Buffer& required_input(const ElementInputs& inputs, std::string_view pad_name)
{
    const auto found = inputs.find(std::string(pad_name));
    if (found == inputs.end() || !found->second) {
        throw std::invalid_argument("ScorePlot requires connected input pad " + std::string(pad_name));
    }
    return *found->second;
}

} // namespace

ElementDescriptor ScorePlot::descriptor()
{
    // Presentation properties (ui-control): the session does not rerun; changes take
    // effect on the next streamed buffer.
    const auto display = PropertyEffect{
        .kind = PropertyEffectKind::UiControl,
        .scope = PropertyInvalidationScope::ElementUi,
    };
    return {
        .type_name = "ScorePlot",
        .klass = "Sink/Plot/Score",
        .purpose = "accumulate attack scores/stats into a stacked score plot (score + confidence panels)",
        .input_pads =
            {
                Pad("sink", PadDirection::Input, Caps(crypto::attack_stats_caps_type)),
            },
        .property_specs =
            {
                PropertySpec("group", std::string("cpa"), "plot comparison group (window)", "", std::monostate{}, "",
                             display),
                PropertySpec("title", std::string(), "plot title", "", std::monostate{}, "", display),
                PropertySpec("metrics", std::vector<std::string>{"score", "relative_margin"},
                             "stacked panels: score plus confidence metrics "
                             "(relative_margin, margin, z_score, robust_z_score, top_k_separation, true_rank)",
                             "", std::monostate{}, "", display),
                PropertySpec("max_units", std::int64_t{64}, "maximum attack units (lines) to plot", "",
                             IntRangeConstraint{1, 1024}, "", display),
                PropertySpec("show_second_score", false,
                             "also plot the second-best score per unit in the score panel (same color as "
                             "the unit's line; needs AttackStats top_k >= 2). Toggle applies live.",
                             "", std::monostate{}, "", display),
            },
        .keywords = {"plot", "score", "cpa", "attack", "sca", "imgui", "implot"},
    };
}

ScorePlot::ScorePlot(std::string name)
    : ScorePlot(std::make_shared<leakflow::plot::ScoreView>(), std::move(name))
{
}

ScorePlot::ScorePlot(std::shared_ptr<leakflow::plot::ScoreView> view, std::string name)
    : Element(std::move(name)), view_(std::move(view))
{
    if (!view_) {
        throw std::invalid_argument("ScorePlot requires a ScoreView");
    }
    configure_from_descriptor(descriptor());
}

void ScorePlot::start()
{
    step_ = 0;
}

void ScorePlot::property_changed(std::string_view name)
{
    // ui-control self-apply: flip the secondary-series display on the live snapshot
    // immediately, so the toggle shows/hides accumulated second scores without a
    // new buffer (works in Idle and mid-run).
    if (name == "show_second_score") {
        view_->set_show_secondary(this->name(), bool_property_or(*this, "show_second_score", false));
    }
}

std::optional<Buffer> ScorePlot::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("ScorePlot requires an input buffer");
    }
    return capture_score(*this, *view_, *input, step_);
}

std::optional<Buffer> ScorePlot::process_inputs(ElementInputs inputs)
{
    return capture_score(*this, *view_, required_input(inputs, "sink"), step_);
}

std::shared_ptr<leakflow::plot::ScoreView> ScorePlot::score_view() const
{
    return view_;
}

void ScorePlot::set_score_view(std::shared_ptr<leakflow::plot::ScoreView> view)
{
    if (!view) {
        throw std::invalid_argument("ScorePlot requires a ScoreView");
    }
    view_ = std::move(view);
}

} // namespace leakflow::plugins::crypto_plot
