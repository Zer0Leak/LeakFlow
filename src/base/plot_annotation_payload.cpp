#include "leakflow/base/plot_annotation_payload.hpp"

#include "leakflow/core/summary_document.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace leakflow::base {
namespace {

void validate_annotation(const PlotAnnotation& annotation)
{
    if (annotation.sample_index < 0) {
        throw std::invalid_argument("PlotAnnotationPayload sample_index values must be non-negative");
    }
    if (annotation.value && !std::isfinite(*annotation.value)) {
        throw std::invalid_argument("PlotAnnotationPayload annotation value must be finite when present");
    }
    if (annotation.norm_value && (!std::isfinite(*annotation.norm_value) || *annotation.norm_value < -1.0
                                  || *annotation.norm_value > 1.0)) {
        throw std::invalid_argument("PlotAnnotationPayload annotation norm_value must be finite and between -1 and 1");
    }
    if (annotation.kind.empty()) {
        throw std::invalid_argument("PlotAnnotationPayload annotation kind cannot be empty");
    }
    for (const auto& [key, value] : annotation.fields) {
        (void)value;
        if (key.empty()) {
            throw std::invalid_argument("PlotAnnotationPayload annotation field keys cannot be empty");
        }
    }
}

void describe_annotation(SummarySection& section, std::size_t index, const PlotAnnotation& annotation)
{
    // Print every PlotAnnotation field, in struct order. sample_index is the headline value;
    // absent optionals render as "none" and the fields collection lists its entries.
    auto& field = section.add_field(
        "annotation[" + std::to_string(index) + "]",
        std::string(),
        SummaryValueRole::Number);
    field.add_child("sample_index", summary_integer(annotation.sample_index), SummaryValueRole::Number);
    field.add_child("value", annotation.value ? std::to_string(*annotation.value) : std::string("none"),
        annotation.value ? SummaryValueRole::Number : SummaryValueRole::Text);
    field.add_child("norm_value", annotation.norm_value ? std::to_string(*annotation.norm_value) : std::string("none"),
        annotation.norm_value ? SummaryValueRole::Number : SummaryValueRole::Text);
    for (const auto& [key, value] : annotation.fields) {
        field.add_child(key, value, SummaryValueRole::Text);
    }
    field.add_child("label", annotation.label, SummaryValueRole::Text);
    field.add_child("text", annotation.text, SummaryValueRole::Text);
    field.add_child("kind", annotation.kind, SummaryValueRole::Text);
    field.add_child("target_index",
        annotation.target_index ? summary_integer(*annotation.target_index) : std::string("none"),
        annotation.target_index ? SummaryValueRole::Number : SummaryValueRole::Text);
}

} // namespace

PlotAnnotationPayload::PlotAnnotationPayload(std::vector<PlotAnnotation> annotations)
    : annotations_(std::move(annotations))
{
    for (const auto& annotation : annotations_) {
        validate_annotation(annotation);
    }
}

std::string PlotAnnotationPayload::type_name() const
{
    return plot_annotation_caps_type;
}

void PlotAnnotationPayload::describe(SummarySection& section, std::int64_t summary_level) const
{
    section.add_field("payload", type_name(), SummaryValueRole::TypeName);
    section.add_field("annotations", summary_integer(static_cast<std::int64_t>(annotations_.size())),
        SummaryValueRole::Number);

    if (annotations_.empty()) {
        return;
    }

    // Levels 1-2 show only the first annotation as a representative sample; level 3 lists every
    // annotation. Level 0 stays at the count only. Each described annotation prints all of its
    // populated fields.
    if (summary_level >= 1) {
        describe_annotation(section, 0, annotations_.front());
    }

    if (summary_level >= 3) {
        for (std::size_t index = 1; index < annotations_.size(); ++index) {
            describe_annotation(section, index, annotations_[index]);
        }
    }
}

const std::vector<PlotAnnotation>& PlotAnnotationPayload::annotations() const
{
    return annotations_;
}

const PlotAnnotation& PlotAnnotationPayload::annotation(std::size_t index) const
{
    return annotations_.at(index);
}

std::size_t PlotAnnotationPayload::annotation_count() const
{
    return annotations_.size();
}

Caps PlotAnnotationPayload::caps() const
{
    return Caps(plot_annotation_caps_type);
}

} // namespace leakflow::base
