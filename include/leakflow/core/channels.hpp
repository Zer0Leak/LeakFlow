#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

// The channels a tensor's channel axis carries -- which leakage model each column
// is (HW(m), HW(y), y(0)...). The channel-axis sibling of Units: where Units labels
// the leading (unit) axis with integers, Channels labels a secondary axis with
// string identities, because channels are inherently named, not indexed. A
// first-class framework concept -- an immutable value on the Buffer envelope, copied
// per branch on a Tee, set by the producing element -- so a later per-channel fusion
// aligns two inputs on channel identity instead of assuming positional agreement.
//
// One grammar serves both the property input and the display; labels are listed in
// column order and never reordered:
//   none              -> "none"        (no channel axis)
//   [HW(y)]           -> one channel
//   [HW(m),HW(y)]     -> two channels, column 0 then column 1
// Bare (bracket-less) text is accepted on parse so metadata such as
// "HW(m),HW(y)" round-trips. Empty / unrecognised text is none.
class Channels {
public:
    Channels() = default; // none

    [[nodiscard]] static Channels none();
    // Column-ordered labels; empty labels are dropped, and an empty result is none.
    [[nodiscard]] static Channels of(std::vector<std::string> labels);
    // Parse the grammar above (with or without the surrounding brackets).
    [[nodiscard]] static Channels parse(std::string_view text);

    [[nodiscard]] bool empty() const noexcept; // true for none
    [[nodiscard]] std::int64_t size() const noexcept; // channel count (columns)
    [[nodiscard]] const std::string& at(std::int64_t index) const; // channel at a column
    [[nodiscard]] const std::vector<std::string>& to_vector() const; // materialise
    [[nodiscard]] std::string format() const; // "none" | "[...]"

    [[nodiscard]] bool operator==(const Channels& other) const;
    [[nodiscard]] bool operator!=(const Channels& other) const { return !(*this == other); }

private:
    std::vector<std::string> labels_;
};

} // namespace leakflow
