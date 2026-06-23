#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

enum class SummaryValueRole {
    Text,
    TypeName,
    CapsType,
    MetadataKey,
    Path,
    Number,
    Size,
    Boolean,
    Warning,
};

struct SummaryValue {
    SummaryValue(std::string text, SummaryValueRole role = SummaryValueRole::Text);

    std::string text;
    SummaryValueRole role = SummaryValueRole::Text;
};

struct SummaryField {
    SummaryField(std::string label, SummaryValue value);
    SummaryField(std::string label, std::string value, SummaryValueRole role = SummaryValueRole::Text);

    SummaryField& add_child(std::string label, SummaryValue value);
    SummaryField& add_child(std::string label, std::string value, SummaryValueRole role = SummaryValueRole::Text);

    std::string label;
    SummaryValue value;
    std::vector<SummaryField> children;
};

struct SummarySection {
    explicit SummarySection(std::string title);

    SummaryField& add_field(std::string label, SummaryValue value);
    SummaryField& add_field(std::string label, std::string value, SummaryValueRole role = SummaryValueRole::Text);

    std::string title;
    std::vector<SummaryField> fields;
};

struct SummaryDocument {
    explicit SummaryDocument(std::string title);

    SummarySection& add_section(std::string title);

    std::string title;
    std::vector<SummarySection> sections;
};

[[nodiscard]] std::string summary_bool(bool value);
[[nodiscard]] std::string summary_integer(std::int64_t value);
[[nodiscard]] std::string summary_size(std::uint64_t value);
[[nodiscard]] std::string summary_list(const std::vector<std::string>& values);
[[nodiscard]] std::string summary_list(const std::vector<std::uint64_t>& values);
[[nodiscard]] std::string summary_list_from_int_array(const std::int64_t* values, std::size_t count);

} // namespace leakflow
