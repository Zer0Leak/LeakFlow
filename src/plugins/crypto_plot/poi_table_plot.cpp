#include "leakflow/plugins/crypto_plot/poi_table_plot.hpp"

#include "leakflow/plot/poi_table_view.hpp"
#include "leakflow/plugins/crypto/correlation_poi_payload.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
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

using leakflow::plugins::crypto::correlation_poi_caps_type;
using leakflow::plugins::crypto::CorrelationPoiPayload;
using leakflow::plugins::crypto::CorrelationPoiResult;

[[nodiscard]] std::shared_ptr<CorrelationPoiPayload> optional_poi(
    const ElementInputs& inputs, std::string_view pad, const Buffer** buffer)
{
    const auto found = inputs.find(std::string(pad));
    if (found == inputs.end() || !found->second) {
        return nullptr;
    }
    if (found->second->caps().type() != correlation_poi_caps_type) {
        throw std::invalid_argument(std::string(pad) + " input must have leakflow/correlation-poi caps");
    }
    auto payload = found->second->payload_as<CorrelationPoiPayload>();
    if (!payload) {
        throw std::invalid_argument(std::string(pad) + " input must carry a CorrelationPoiPayload");
    }
    if (*buffer == nullptr) {
        *buffer = &*found->second;
    }
    return payload;
}

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

[[nodiscard]] const CorrelationPoiResult* result_for(const CorrelationPoiPayload* payload, std::int64_t unit_index)
{
    if (payload == nullptr) {
        return nullptr;
    }
    for (const auto& result : payload->results()) {
        if (result.unit_index == unit_index) {
            return &result;
        }
    }
    return nullptr;
}

[[nodiscard]] std::string format_score(double value, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

// One (unit, channel) score column as raw numbers, plus whether the source actually contributed.
// Missing input, or a shape that does not match the reference (channels, k), yields NaNs so the
// view shows "-", skips the line, and never highlights it.
struct ScoreColumn {
    std::vector<double> values;
    bool present = false;
};

[[nodiscard]] ScoreColumn score_column(const CorrelationPoiResult* result, std::int64_t channel, std::int64_t k)
{
    if (result == nullptr || channel >= result->result.size(0) || result->result.size(1) != k) {
        return {std::vector<double>(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN()), false};
    }
    const auto last = result->result.size(2) - 1; // (index, score) -> score; (score) -> score
    const auto scores = result->result[channel].select(1, last).to(torch::kFloat64).to(torch::kCPU).contiguous();
    const auto accessor = scores.accessor<double, 1>();
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(k));
    for (std::int64_t index = 0; index < k; ++index) {
        values.push_back(accessor[index]);
    }
    return {std::move(values), true};
}

[[nodiscard]] std::vector<std::string> format_column(const ScoreColumn& column, int precision)
{
    std::vector<std::string> row;
    row.reserve(column.values.size());
    for (const auto value : column.values) {
        row.push_back(column.present ? format_score(value, precision) : std::string("-"));
    }
    return row;
}

// Numeric column keys: PoI sample indexes when the payload carries them (last axis >= 2), else 1..k.
[[nodiscard]] std::vector<double> column_samples(const CorrelationPoiResult& primary, std::int64_t channel)
{
    const auto k = primary.result.size(1);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(k));
    if (primary.result.size(2) >= 2) {
        const auto indexes = primary.result[channel].select(1, 0).to(torch::kFloat64).to(torch::kCPU).contiguous();
        const auto accessor = indexes.accessor<double, 1>();
        for (std::int64_t index = 0; index < k; ++index) {
            samples.push_back(accessor[index]);
        }
    } else {
        for (std::int64_t index = 0; index < k; ++index) {
            samples.push_back(static_cast<double>(index + 1));
        }
    }
    return samples;
}

[[nodiscard]] std::vector<std::string> headers_from_samples(const std::vector<double>& samples)
{
    std::vector<std::string> headers;
    headers.reserve(samples.size());
    for (const auto sample : samples) {
        headers.push_back(std::to_string(static_cast<std::int64_t>(std::llround(sample))));
    }
    return headers;
}

[[nodiscard]] std::vector<std::string> channel_labels(const Buffer* buffer, std::int64_t channel_count)
{
    std::vector<std::string> labels;
    if (buffer != nullptr && buffer->has_metadata("payload.leakage.channels")) {
        auto text = buffer->metadata("payload.leakage.channels");
        std::size_t begin = 0;
        for (std::size_t index = 0; index <= text.size(); ++index) {
            if (index == text.size() || text[index] == ',') {
                auto token = text.substr(begin, index - begin);
                const auto first = token.find_first_not_of(" \t");
                const auto tlast = token.find_last_not_of(" \t");
                labels.push_back(first == std::string::npos ? "" : token.substr(first, tlast - first + 1));
                begin = index + 1;
            }
        }
    }
    if (static_cast<std::int64_t>(labels.size()) != channel_count) {
        labels.clear();
        for (std::int64_t channel = 0; channel < channel_count; ++channel) {
            labels.push_back("channel " + std::to_string(channel));
        }
    }
    return labels;
}

} // namespace

ElementDescriptor PoiTablePlot::descriptor()
{
    return {
        .type_name = "PoiTablePlot",
        .klass = "Sink/Plot/Table/PoI",
        .purpose = "compare a reference PoI set against a current one in a per-(unit,channel) table",
        .input_pads = {
            Pad("reference", PadDirection::Input, Caps(correlation_poi_caps_type), PadPresence::Optional),
            Pad("current", PadDirection::Input, Caps(correlation_poi_caps_type), PadPresence::Optional),
        },
        .property_specs = {
            PropertySpec("reference_label", std::string("reference"), "row label for the reference input",
                "", std::monostate{}, "", PropertyEffect{}),
            PropertySpec("current_label", std::string("current"), "row label for the current input",
                "", std::monostate{}, "", PropertyEffect{}),
            PropertySpec("title", std::string("PoI comparison"), "window title", "", std::monostate{}, "",
                PropertyEffect{}),
            PropertySpec("precision", std::int64_t{3}, "displayed correlation precision", "",
                IntRangeConstraint{0, 12}, "", PropertyEffect{}),
        },
        .keywords = {"poi", "table", "compare", "correlation", "plot"},
    };
}

PoiTablePlot::PoiTablePlot(std::string name)
    : PoiTablePlot(std::make_shared<leakflow::plot::PoiTableView>(), std::move(name))
{
}

PoiTablePlot::PoiTablePlot(std::shared_ptr<leakflow::plot::PoiTableView> view, std::string name)
    : Element(std::move(name)), view_(std::move(view))
{
    if (!view_) {
        throw std::invalid_argument("PoiTablePlot requires a PoiTableView");
    }
    configure_from_descriptor(descriptor());
}

std::optional<Buffer> PoiTablePlot::process(std::optional<Buffer>)
{
    throw std::invalid_argument("PoiTablePlot requires named reference and/or current inputs");
}

std::optional<Buffer> PoiTablePlot::process_inputs(ElementInputs inputs)
{
    const Buffer* primary_buffer = nullptr;
    const auto reference = optional_poi(inputs, "reference", &primary_buffer);
    const auto current = optional_poi(inputs, "current", &primary_buffer);
    if (!reference && !current) {
        throw std::invalid_argument("PoiTablePlot needs at least one of reference / current");
    }
    const auto& primary = reference ? *reference : *current;
    if (primary.results().empty()) {
        throw std::invalid_argument("PoiTablePlot: PoI payload has no units");
    }

    const auto precision = static_cast<int>(int_property_or(*this, "precision", 3));
    const auto channels = primary.results().front().result.size(0);
    const auto labels = channel_labels(primary_buffer, channels);

    std::vector<std::int64_t> unit_ids;
    std::vector<leakflow::plot::PoiTableGroup> groups; // unit-major: [unit * channels + channel]
    for (const auto& base_result : primary.results()) {
        unit_ids.push_back(base_result.unit_index);
        const auto* reference_result = result_for(reference.get(), base_result.unit_index);
        const auto* current_result = result_for(current.get(), base_result.unit_index);
        const auto k = base_result.result.size(1);
        for (std::int64_t channel = 0; channel < channels; ++channel) {
            const auto samples = column_samples(base_result, channel);
            auto reference_column = score_column(reference_result, channel, k);
            auto current_column = score_column(current_result, channel, k);
            groups.push_back(leakflow::plot::PoiTableGroup{
                .columns = headers_from_samples(samples),
                .reference = format_column(reference_column, precision),
                .current = format_column(current_column, precision),
                .sample = samples,
                .reference_values = std::move(reference_column.values),
                .current_values = std::move(current_column.values),
                .has_reference = reference_column.present,
                .has_current = current_column.present,
            });
        }
    }

    view_->set_table(name(), string_property_or(*this, "title", "PoI comparison"),
        string_property_or(*this, "reference_label", "reference"),
        string_property_or(*this, "current_label", "current"), std::move(unit_ids), labels, std::move(groups));
    return std::nullopt; // sink
}

std::shared_ptr<leakflow::plot::PoiTableView> PoiTablePlot::poi_table_view() const
{
    return view_;
}

} // namespace leakflow::plugins::crypto_plot
