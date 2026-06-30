#include "leakflow/core/buffer.hpp"

#include <cstdint>
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
    } else {
        payload_section.add_field("payload", "none");
    }

    return document;
}

} // namespace leakflow
