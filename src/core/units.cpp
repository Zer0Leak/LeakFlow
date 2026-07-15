#include "leakflow/core/units.hpp"

#include <cctype>
#include <charconv>
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

// Parse a non-negative integer from trimmed text; returns false on any leftover.
[[nodiscard]] bool parse_int(std::string_view text, std::int64_t& value)
{
    text = trim(text);
    if (text.empty()) {
        return false;
    }
    const auto* first = text.data();
    const auto* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    return result.ec == std::errc{} && result.ptr == last;
}

// Whether values is a contiguous ascending run (v[i] == v[0] + i).
[[nodiscard]] bool is_contiguous_ascending(const std::vector<std::int64_t>& values)
{
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] != values[index - 1] + 1) {
            return false;
        }
    }
    return true;
}

} // namespace

Units Units::none()
{
    return {};
}

Units Units::range(std::int64_t start, std::int64_t count)
{
    Units labels;
    if (count > 0) {
        labels.start_ = start;
        labels.count_ = count;
    }
    return labels;
}

Units Units::of(std::vector<std::int64_t> values)
{
    if (values.empty()) {
        return {};
    }
    if (is_contiguous_ascending(values)) {
        return range(values.front(), static_cast<std::int64_t>(values.size()));
    }
    Units labels;
    labels.explicit_ = std::move(values);
    return labels;
}

Units Units::parse(std::string_view text)
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

    std::vector<std::int64_t> values;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto comma = text.find(',', start);
        const auto piece = trim(text.substr(start, comma == std::string_view::npos ? comma : comma - start));
        if (!piece.empty()) {
            const auto colon = piece.find(':');
            if (colon == std::string_view::npos) {
                std::int64_t single = 0;
                if (parse_int(piece, single)) {
                    values.push_back(single);
                }
            } else {
                std::int64_t low = 0;
                std::int64_t high = 0; // exclusive
                if (parse_int(piece.substr(0, colon), low) && parse_int(piece.substr(colon + 1), high)) {
                    for (std::int64_t value = low; value < high; ++value) {
                        values.push_back(value);
                    }
                }
            }
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return of(std::move(values));
}

bool Units::empty() const noexcept
{
    return count_ == 0 && explicit_.empty();
}

std::int64_t Units::size() const noexcept
{
    return explicit_.empty() ? count_ : static_cast<std::int64_t>(explicit_.size());
}

std::int64_t Units::at(std::int64_t index) const
{
    if (!explicit_.empty()) {
        return explicit_[static_cast<std::size_t>(index)];
    }
    return start_ + index;
}

std::vector<std::int64_t> Units::to_vector() const
{
    if (!explicit_.empty()) {
        return explicit_;
    }
    std::vector<std::int64_t> values;
    values.reserve(static_cast<std::size_t>(count_));
    for (std::int64_t offset = 0; offset < count_; ++offset) {
        values.push_back(start_ + offset);
    }
    return values;
}

std::string Units::format() const
{
    if (empty()) {
        return "none";
    }
    if (explicit_.empty()) {
        if (count_ == 1) {
            return "[" + std::to_string(start_) + "]";
        }
        return "[" + std::to_string(start_) + ":" + std::to_string(start_ + count_) + "]";
    }

    // Explicit: collapse maximal ascending-consecutive runs (a:b, b exclusive,
    // Python-slice style), keeping row order so a reordered set prints as written.
    std::string items;
    std::size_t index = 0;
    while (index < explicit_.size()) {
        std::size_t last = index;
        while (last + 1 < explicit_.size() && explicit_[last + 1] == explicit_[last] + 1) {
            ++last;
        }
        if (!items.empty()) {
            items += ",";
        }
        if (last == index) {
            items += std::to_string(explicit_[index]);
        } else {
            items += std::to_string(explicit_[index]) + ":" + std::to_string(explicit_[last] + 1);
        }
        index = last + 1;
    }
    return "[" + items + "]";
}

bool Units::operator==(const Units& other) const
{
    return start_ == other.start_ && count_ == other.count_ && explicit_ == other.explicit_;
}

} // namespace leakflow
