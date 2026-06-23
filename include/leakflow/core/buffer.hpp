#pragma once

#include "leakflow/core/caps.hpp"
#include "leakflow/core/payload.hpp"
#include "leakflow/core/summary_document.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

class Buffer {
public:
    explicit Buffer(Caps caps);

    [[nodiscard]] const Caps& caps() const;
    [[nodiscard]] Caps& caps();

    [[nodiscard]] bool has_metadata(std::string_view key) const;
    [[nodiscard]] const std::string& metadata(std::string_view key) const;
    [[nodiscard]] std::string metadata_or(std::string_view key, std::string default_value) const;

    void set_metadata(std::string key, std::string value);

    [[nodiscard]] const std::map<std::string, std::string>& metadata() const;

    [[nodiscard]] bool has_payload() const;
    [[nodiscard]] std::shared_ptr<Payload> payload() const;
    [[nodiscard]] bool payload_is_unique() const;

    void set_payload(std::shared_ptr<Payload> payload);

    // Vector-clock buffer provenance (Phase 27). Dense per-element production
    // counts; the executor is the single writer (sets it on every routed buffer).
    // Empty on freshly constructed buffers. See
    // docs/design/dataflow_sync_model.md Section 6.
    [[nodiscard]] const std::vector<std::uint32_t>& provenance() const;
    void set_provenance(std::vector<std::uint32_t> provenance);

    [[nodiscard]] SummaryDocument describe(std::int64_t summary_level) const;

    template <typename T>
    [[nodiscard]] std::shared_ptr<T> payload_as() const
    {
        return std::dynamic_pointer_cast<T>(payload_);
    }

    template <typename T>
    [[nodiscard]] T* mutable_payload_if_unique()
    {
        if (!payload_is_unique()) {
            return nullptr;
        }

        return dynamic_cast<T*>(payload_.get());
    }

private:
    Caps caps_;
    std::map<std::string, std::string> metadata_;
    std::shared_ptr<Payload> payload_;
    std::vector<std::uint32_t> provenance_;
};

} // namespace leakflow
