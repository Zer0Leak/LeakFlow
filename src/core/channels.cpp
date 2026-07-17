#include "leakflow/core/channels.hpp"

#include <cctype>
#include <utility>

namespace leakflow {
namespace {

[[nodiscard]] std::string_view trim(std::string_view text)
{
    while (!text.empty() && (std::isspace(static_cast<unsigned char>(text.front())) != 0)) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (std::isspace(static_cast<unsigned char>(text.back())) != 0)) {
        text.remove_suffix(1);
    }
    return text;
}

} // namespace

Channels Channels::none()
{
    return {};
}

Channels Channels::of(std::vector<std::string> labels)
{
    std::vector<std::string> kept;
    kept.reserve(labels.size());
    for (auto& label : labels) {
        if (!label.empty()) {
            kept.push_back(std::move(label));
        }
    }
    Channels channels;
    channels.labels_ = std::move(kept);
    return channels;
}

Channels Channels::parse(std::string_view text)
{
    text = trim(text);
    if (text.empty() || text == "none") {
        return {};
    }
    if (text.front() == '[' && text.back() == ']') {
        text.remove_prefix(1);
        text.remove_suffix(1);
        text = trim(text);
    }
    if (text.empty()) {
        return {};
    }

    // Channel names contain '(' and ')' (HW(m), y(0)) but never a comma, so a plain
    // comma split is unambiguous.
    std::vector<std::string> labels;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto comma = text.find(',', start);
        const auto piece = trim(text.substr(start, comma == std::string_view::npos ? comma : comma - start));
        if (!piece.empty()) {
            labels.emplace_back(piece);
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return of(std::move(labels));
}

bool Channels::empty() const noexcept
{
    return labels_.empty();
}

std::int64_t Channels::size() const noexcept
{
    return static_cast<std::int64_t>(labels_.size());
}

const std::string& Channels::at(std::int64_t index) const
{
    return labels_.at(static_cast<std::size_t>(index));
}

const std::vector<std::string>& Channels::to_vector() const
{
    return labels_;
}

std::string Channels::format() const
{
    if (labels_.empty()) {
        return "none";
    }
    std::string items;
    for (std::size_t index = 0; index < labels_.size(); ++index) {
        if (index != 0) {
            items += ",";
        }
        items += labels_[index];
    }
    return "[" + items + "]";
}

bool Channels::operator==(const Channels& other) const
{
    return labels_ == other.labels_;
}

} // namespace leakflow
