#include "leakflow/plugins/crypto_plot/score_table_plot.hpp"

#include "leakflow/plugins/crypto/attack_payload.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
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

[[nodiscard]] std::shared_ptr<crypto::AttackStatsPayload> stats_payload_for(const Buffer& input)
{
    if (input.caps().type() != crypto::attack_stats_caps_type) {
        throw std::invalid_argument("ScoreTablePlot requires leakflow/attack-stats input caps");
    }
    auto payload = input.payload_as<crypto::AttackStatsPayload>();
    if (!payload) {
        throw std::invalid_argument("ScoreTablePlot requires an AttackStatsPayload");
    }
    return payload;
}

// N for this buffer: the cumulative observation count stamped by CpaAttack, or a
// running step index when that metadata is absent.
[[nodiscard]] std::int64_t observation_n(const Buffer& input, std::int64_t& step)
{
    if (input.has_metadata("attack.observation_count")) {
        const auto& text = input.metadata("attack.observation_count");
        std::size_t consumed = 0;
        const auto value = std::stoll(text, &consumed);
        if (consumed == text.size() && value > 0) {
            return value;
        }
    }
    return ++step;
}

[[nodiscard]] std::string unit_label(const std::vector<std::int64_t>& units, std::int64_t unit)
{
    if (unit >= 0 && static_cast<std::size_t>(unit) < units.size()) {
        return std::to_string(units[static_cast<std::size_t>(unit)]);
    }
    return std::to_string(unit);
}

[[nodiscard]] std::string format_score(double value)
{
    // Fixed decimals so every score has the same width and the column stays aligned
    // (e.g. 0.3450 and 0.7192, not 0.345 and 0.7192).
    std::ostringstream output;
    output.setf(std::ios::fixed, std::ios::floatfield);
    output.precision(4);
    output << value;
    return output.str();
}

[[nodiscard]] std::string format_guess(std::int64_t guess)
{
    std::ostringstream output;
    output << std::hex;
    if (guess >= 0 && guess <= 0xFF) {
        output.width(2);
        output.fill('0');
    }
    output << guess;
    return output.str();
}

std::optional<Buffer> capture_table(Element& element, leakflow::plot::TableView& view, const Buffer& input,
                                    std::int64_t& step)
{
    const auto payload = stats_payload_for(input);
    const auto group = string_property_or(element, "group", "cpa");
    const auto title = string_property_or(element, "title", "");
    const auto sort = string_property_or(element, "sort", "score");
    const bool sort_by_guess = sort == "guess";
    const auto max_units = std::max<std::int64_t>(integer_property_or(element, "max_units", 16), 1);
    const auto max_rows = std::max<std::int64_t>(integer_property_or(element, "rows", 16), 1);
    const auto max_history = std::max<std::int64_t>(integer_property_or(element, "max_history", 1), 0);

    const auto n = observation_n(input, step);
    const auto unit_count = std::min(payload->unit_count(), max_units);
    const auto rank_count = std::min(payload->top_k(), max_rows);
    const bool has_truth = payload->has_truth();

    const auto topk_guess = payload->topk_guess().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto topk_score = payload->topk_score().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto true_guess = has_truth ? payload->true_guess()->to(torch::kCPU).to(torch::kInt64).contiguous()
                                      : torch::Tensor{};
    const auto true_rank = has_truth ? payload->true_rank()->to(torch::kCPU).to(torch::kInt64).contiguous()
                                     : torch::Tensor{};
    const auto success = has_truth ? payload->success()->to(torch::kCPU).to(torch::kBool).contiguous()
                                   : torch::Tensor{};
    const auto& units = payload->units();

    // Correct-key highlight (green); leader (best-by-score) is emphasized in white.
    static constexpr leakflow::plot::TracePlotColor correct_key_tint{0.10F, 0.55F, 0.20F};

    leakflow::plot::TableUpdate update;
    update.group = group;
    update.title = title;
    update.max_history = static_cast<std::size_t>(max_history);
    update.frame.n = n;
    update.frame.caption = "N = " + std::to_string(n);

    update.columns.reserve(static_cast<std::size_t>(unit_count) + 1);
    update.columns.emplace_back("rank/unit"); // corner: rows are ranks, columns are units
    for (std::int64_t unit = 0; unit < unit_count; ++unit) {
        update.columns.emplace_back(unit_label(units, unit));
    }

    update.frame.rows.assign(static_cast<std::size_t>(rank_count),
                             std::vector<leakflow::plot::TableCell>(static_cast<std::size_t>(unit_count) + 1));
    for (std::int64_t rank = 0; rank < rank_count; ++rank) {
        update.frame.rows[static_cast<std::size_t>(rank)][0].text = "#" + std::to_string(rank + 1);
    }

    for (std::int64_t unit = 0; unit < unit_count; ++unit) {
        // Position order within this unit column: score-ranked as-is, or re-sorted by
        // guess value (each column independently, so the row is a position index).
        std::vector<std::int64_t> order(static_cast<std::size_t>(rank_count));
        std::iota(order.begin(), order.end(), std::int64_t{0});
        if (sort_by_guess) {
            std::stable_sort(order.begin(), order.end(), [&](std::int64_t left, std::int64_t right) {
                return topk_guess[unit][left].item<std::int64_t>() < topk_guess[unit][right].item<std::int64_t>();
            });
        }

        const auto truth_guess = has_truth ? true_guess[unit].item<std::int64_t>() : std::int64_t{-1};
        const auto truth_rank = has_truth ? true_rank[unit].item<std::int64_t>() : std::int64_t{-1};
        const bool truth_ok = has_truth && success[unit].item<bool>();

        for (std::int64_t rank = 0; rank < rank_count; ++rank) {
            const auto position = order[static_cast<std::size_t>(rank)];
            const auto guess = topk_guess[unit][position].item<std::int64_t>();
            const auto score = topk_score[unit][position].item<double>();
            const bool correct = has_truth && guess == truth_guess;

            leakflow::plot::TableCell cell;
            cell.text = format_guess(guess) + "  " + format_score(score);
            cell.emphasize = position == 0; // best-by-score candidate for this unit
            if (correct) {
                cell.tint = correct_key_tint;
            }
            cell.hover.emplace_back("unit", unit_label(units, unit));
            cell.hover.emplace_back("guess", "0x" + format_guess(guess));
            cell.hover.emplace_back("score", format_score(score));
            cell.hover.emplace_back("rank", std::to_string(position + 1));
            if (has_truth) {
                cell.hover.emplace_back("correct key", correct ? "yes" : "no");
                cell.hover.emplace_back("unit true rank", std::to_string(truth_rank));
                cell.hover.emplace_back("unit success", truth_ok ? "true" : "false");
            }
            update.frame.rows[static_cast<std::size_t>(rank)][static_cast<std::size_t>(unit) + 1] = std::move(cell);
        }
    }

    view.push(element.name(), update);

    auto record = element.make_log_record(log::LogLevel::Debug, "element", "captured score-table frame");
    record.fields.emplace("units", std::to_string(unit_count));
    record.fields.emplace("rows", std::to_string(rank_count));
    record.fields.emplace("N", std::to_string(n));
    leakflow::log::write(std::move(record));

    return std::nullopt; // sink: the scoreboard view owns the snapshot
}

[[nodiscard]] const Buffer& required_input(const ElementInputs& inputs, std::string_view pad_name)
{
    const auto found = inputs.find(std::string(pad_name));
    if (found == inputs.end() || !found->second) {
        throw std::invalid_argument("ScoreTablePlot requires connected input pad " + std::string(pad_name));
    }
    return *found->second;
}

} // namespace

ElementDescriptor ScoreTablePlot::descriptor()
{
    // Presentation properties (ui-control): the session does not rerun; changes take
    // effect on the next streamed buffer.
    const auto display = PropertyEffect{
        .kind = PropertyEffectKind::UiControl,
        .scope = PropertyInvalidationScope::ElementUi,
    };
    return {
        .type_name = "ScoreTablePlot",
        .klass = "Sink/Plot/AttackScoreboard",
        .purpose = "render attack scores as a ranked candidate table (units as columns, guesses as rows)",
        .input_pads =
            {
                Pad("sink", PadDirection::Input, Caps(crypto::attack_stats_caps_type)),
            },
        .property_specs =
            {
                PropertySpec("group", std::string("cpa"), "plot comparison group (window)", "", std::monostate{}, "",
                             display),
                PropertySpec("title", std::string(), "plot title", "", std::monostate{}, "", display),
                PropertySpec("sort", std::string("score"),
                             "row order per unit column: score (ranked, descending) or guess (by guess value)", "",
                             std::monostate{}, "", display),
                PropertySpec("max_units", std::int64_t{16}, "maximum attack units (columns) to show", "",
                             IntRangeConstraint{1, 256}, "", display),
                PropertySpec("rows", std::int64_t{16}, "maximum ranked rows to show (capped at AttackStats top_k)", "",
                             IntRangeConstraint{1, 256}, "", display),
                PropertySpec("max_history", std::int64_t{1},
                             "kept snapshots (an N-scrubber): 1 = replace (current only), N = keep last N, "
                             "0 = unbounded",
                             "", IntRangeConstraint{0, 100000}, "", display),
            },
        .keywords = {"plot", "score", "table", "cpa", "attack", "sca", "imgui", "ranking"},
    };
}

ScoreTablePlot::ScoreTablePlot(std::string name)
    : ScoreTablePlot(std::make_shared<leakflow::plot::TableView>(), std::move(name))
{
}

ScoreTablePlot::ScoreTablePlot(std::shared_ptr<leakflow::plot::TableView> view, std::string name)
    : Element(std::move(name)), view_(std::move(view))
{
    if (!view_) {
        throw std::invalid_argument("ScoreTablePlot requires a TableView");
    }
    configure_from_descriptor(descriptor());
}

void ScoreTablePlot::start()
{
    step_ = 0;
}

std::optional<Buffer> ScoreTablePlot::process(std::optional<Buffer> input)
{
    if (!input) {
        throw std::invalid_argument("ScoreTablePlot requires an input buffer");
    }
    return capture_table(*this, *view_, *input, step_);
}

std::optional<Buffer> ScoreTablePlot::process_inputs(ElementInputs inputs)
{
    return capture_table(*this, *view_, required_input(inputs, "sink"), step_);
}

std::shared_ptr<leakflow::plot::TableView> ScoreTablePlot::table_view() const
{
    return view_;
}

void ScoreTablePlot::set_table_view(std::shared_ptr<leakflow::plot::TableView> view)
{
    if (!view) {
        throw std::invalid_argument("ScoreTablePlot requires a TableView");
    }
    view_ = std::move(view);
}

} // namespace leakflow::plugins::crypto_plot
