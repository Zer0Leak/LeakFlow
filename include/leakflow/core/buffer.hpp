#pragma once

#include "leakflow/core/units.hpp"
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

    // The units the payload's leading axis carries -- which unit each row is (see
    // Units: none / range / explicit). An immutable value member, copied per buffer on
    // a Tee fan-out, set by the producing element alongside the payload so a later
    // fusion can verify two inputs describe the same units, not merely the same shape.
    // A first-class framework concept, so it lives on the buffer, not inside the payload.
    [[nodiscard]] const Units& units() const;
    void set_units(Units units);

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
    Units units_;
};

// Alignment of two leading-axis label vectors by value: the labels they share and
// the axis-0 positions of each shared label in each input, so a caller can
// index_select both down to a common leading axis. `shared` is in `a` order.
// `identical` is true when a and b are the same labels in the same order (the fast
// path, no reindex needed). A domain fusion reads this as "the shared units".
struct LabelAlignment {
    std::vector<std::int64_t> shared;
    std::vector<std::int64_t> a_indices;
    std::vector<std::int64_t> b_indices;
    bool identical = false;
};

[[nodiscard]] LabelAlignment align_labels(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b);

} // namespace leakflow
