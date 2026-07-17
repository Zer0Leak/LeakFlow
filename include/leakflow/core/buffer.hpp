#pragma once

#include "leakflow/core/units.hpp"
#include "leakflow/core/channels.hpp"
#include "leakflow/core/caps.hpp"
#include "leakflow/core/payload.hpp"
#include "leakflow/core/summary_document.hpp"

#include <cstddef>
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

    // The channels the payload's channel axis carries -- which leakage model each
    // column is (see Channels). The channel-axis sibling of units(): an immutable
    // value member, copied per buffer on a Tee fan-out, set by the producing element
    // so a later per-channel fusion aligns on channel identity, not column position.
    [[nodiscard]] const Channels& channels() const;
    void set_channels(Channels channels);

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
    Channels channels_;
};

// Alignment of two labelled axes by value: the labels they share and the axis
// positions of each shared label in each input, so a caller can index_select both
// down to a common axis. `shared` is in `a` order. `identical` is true when a and b
// are the same labels in the same order (the fast path, no reindex needed). A domain
// fusion reads this as "the shared units" (int labels) or "the shared channels"
// (string labels) -- one mechanism for every labelled semantic axis.
template <typename Label>
struct Alignment {
    std::vector<Label> shared;
    std::vector<std::int64_t> a_indices;
    std::vector<std::int64_t> b_indices;
    bool identical = false;
};

using LabelAlignment = Alignment<std::int64_t>;
using ChannelAlignment = Alignment<std::string>;

// Label counts are tiny (a handful of attack bytes/channels), so a direct scan is
// both simplest and fastest. Matching is by value, so a and b may list the shared
// labels in different orders.
template <typename Label>
[[nodiscard]] Alignment<Label> align_labels(const std::vector<Label>& a, const std::vector<Label>& b)
{
    Alignment<Label> alignment;
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
