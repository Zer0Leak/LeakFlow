#include "leakflow/core/buffer.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace leakflow {

Buffer::Buffer(Caps caps)
    : caps_(std::move(caps))
{
}

const Caps& Buffer::caps() const
{
    return caps_;
}

Caps& Buffer::caps()
{
    return caps_;
}

bool Buffer::has_metadata(std::string_view key) const
{
    return metadata_.contains(std::string(key));
}

const std::string& Buffer::metadata(std::string_view key) const
{
    return metadata_.at(std::string(key));
}

std::string Buffer::metadata_or(std::string_view key, std::string default_value) const
{
    const auto found = metadata_.find(std::string(key));
    if (found == metadata_.end()) {
        return default_value;
    }

    return found->second;
}

void Buffer::set_metadata(std::string key, std::string value)
{
    metadata_[std::move(key)] = std::move(value);
}

const std::map<std::string, std::string>& Buffer::metadata() const
{
    return metadata_;
}

bool Buffer::has_payload() const
{
    return payload_ != nullptr;
}

std::shared_ptr<Payload> Buffer::payload() const
{
    return payload_;
}

bool Buffer::payload_is_unique() const
{
    return payload_ != nullptr && payload_.use_count() == 1;
}

void Buffer::set_payload(std::shared_ptr<Payload> payload)
{
    if (!payload) {
        payload_.reset();
        metadata_.erase("payload.layout");
        return;
    }

    auto layout = payload->layout();
    if (layout.empty()) {
        throw std::invalid_argument("payload layout must not be empty");
    }

    metadata_.insert_or_assign("payload.layout", std::move(layout));
    payload_ = std::move(payload);
}

const std::vector<std::uint32_t>& Buffer::provenance() const
{
    return provenance_;
}

void Buffer::set_provenance(std::vector<std::uint32_t> provenance)
{
    provenance_ = std::move(provenance);
}

const Units& Buffer::units() const
{
    return units_;
}

void Buffer::set_units(Units units)
{
    units_ = std::move(units);
}

SummaryDocument Buffer::describe(std::int64_t summary_level) const
{
    SummaryDocument document("Buffer");

    auto& overview = document.add_section("Overview");
    overview.add_field("caps", caps_.to_string(), SummaryValueRole::CapsType);
    overview.add_field("metadata", summary_size(metadata_.size()), SummaryValueRole::Number);

    if (summary_level >= 2 && !caps_.params().empty()) {
        auto& caps_section = document.add_section("Caps");
        for (const auto& [key, value] : caps_.params()) {
            caps_section.add_field(key, value);
        }
    }

    if (summary_level >= 2 && !metadata_.empty()) {
        auto& metadata_section = document.add_section("Metadata");
        for (const auto& [key, value] : metadata_) {
            if (summary_level >= 3) {
                const auto role = key.find("path") != std::string::npos ? SummaryValueRole::Path : SummaryValueRole::Text;
                metadata_section.add_field(key, value, role);
            } else {
                metadata_section.add_field("key", key, SummaryValueRole::MetadataKey);
            }
        }
    }

    auto& payload_section = document.add_section("Payload");
    if (payload_) {
        payload_->describe(payload_section, summary_level);
        if (!units_.empty()) {
            // Name the field after the leading axis from payload.layout, pluralised
            // ("unit/trace" -> "units"), so core prints the domain's word for the
            // axis without hardcoding it. The value is the shared Units form.
            std::string axis_name = "axis";
            if (const auto found = metadata_.find("payload.layout"); found != metadata_.end()) {
                const auto& layout = found->second;
                const auto slash = layout.find('/');
                axis_name = slash == std::string::npos ? layout : layout.substr(0, slash);
            }
            if (!axis_name.empty() && axis_name.back() != 's') {
                axis_name += 's';
            }
            payload_section.add_field(axis_name, units_.format());
        }
    } else {
        payload_section.add_field("payload", "none");
    }

    return document;
}

LabelAlignment align_labels(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b)
{
    // Unit counts are tiny (a handful of attack bytes), so a direct scan is both
    // simplest and fastest. Matching is by value, so a and b may list the shared
    // labels in different orders.
    LabelAlignment alignment;
    for (std::size_t ai = 0; ai < a.size(); ++ai) {
        for (std::size_t bi = 0; bi < b.size(); ++bi) {
            if (a[ai] == b[bi]) {
                alignment.shared.push_back(a[ai]);
                alignment.a_indices.push_back(static_cast<std::int64_t>(ai));
                alignment.b_indices.push_back(static_cast<std::int64_t>(bi));
                break;
            }
        }
    }
    alignment.identical = (a == b);
    return alignment;
}

} // namespace leakflow
