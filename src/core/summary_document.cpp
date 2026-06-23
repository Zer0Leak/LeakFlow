#include "leakflow/core/summary_document.hpp"

#include <sstream>
#include <utility>

namespace leakflow {
namespace {

template <typename T>
[[nodiscard]] std::string bracketed_list(const std::vector<T>& values)
{
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
    return output.str();
}

} // namespace

SummaryValue::SummaryValue(std::string text, SummaryValueRole role)
    : text(std::move(text))
    , role(role)
{
}

SummaryField::SummaryField(std::string label, SummaryValue value)
    : label(std::move(label))
    , value(std::move(value))
{
}

SummaryField::SummaryField(std::string label, std::string value, SummaryValueRole role)
    : SummaryField(std::move(label), SummaryValue(std::move(value), role))
{
}

SummaryField& SummaryField::add_child(std::string label, SummaryValue value)
{
    children.emplace_back(std::move(label), std::move(value));
    return children.back();
}

SummaryField& SummaryField::add_child(std::string label, std::string value, SummaryValueRole role)
{
    return add_child(std::move(label), SummaryValue(std::move(value), role));
}

SummarySection::SummarySection(std::string title)
    : title(std::move(title))
{
}

SummaryField& SummarySection::add_field(std::string label, SummaryValue value)
{
    fields.emplace_back(std::move(label), std::move(value));
    return fields.back();
}

SummaryField& SummarySection::add_field(std::string label, std::string value, SummaryValueRole role)
{
    return add_field(std::move(label), SummaryValue(std::move(value), role));
}

SummaryDocument::SummaryDocument(std::string title)
    : title(std::move(title))
{
}

SummarySection& SummaryDocument::add_section(std::string title)
{
    sections.emplace_back(std::move(title));
    return sections.back();
}

std::string summary_bool(bool value)
{
    return value ? "true" : "false";
}

std::string summary_integer(std::int64_t value)
{
    return std::to_string(value);
}

std::string summary_size(std::uint64_t value)
{
    return std::to_string(value);
}

std::string summary_list(const std::vector<std::string>& values)
{
    return bracketed_list(values);
}

std::string summary_list(const std::vector<std::uint64_t>& values)
{
    return bracketed_list(values);
}

std::string summary_list_from_int_array(const std::int64_t* values, std::size_t count)
{
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
    return output.str();
}

} // namespace leakflow
