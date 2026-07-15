#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace leakflow {

// The units a tensor's leading axis carries -- which AES byte / Kyber coefficient
// each row is. A first-class framework concept (both crypto and ml batch over a unit
// axis), immutable after construction, in one of three forms:
//   none                        -- no unit axis (raw data, "just the numbers")
//   range [start, start+count)  -- the common case: one unit [x], a contiguous run,
//                                  or the whole axis [0-N] materialised from size(0)
//   explicit v0,v1,...          -- an arbitrary / gappy / reordered set (row i is v[i])
//
// One grammar serves both the property input and the display; ranges are
// Python-slice style (a:b, b exclusive):
//   none   -> "none"
//   single -> "[0]"
//   range  -> "[0:16]"      (the second number is exclusive, so [0:16] is 0..15)
//   set    -> "[0,1,4:7]"   (commas separate items; a:b is a run, b exclusive)
class Units {
public:
    Units() = default; // none

    [[nodiscard]] static Units none();
    // A contiguous run [start, start+count). count <= 0 collapses to none.
    [[nodiscard]] static Units range(std::int64_t start, std::int64_t count);
    // Row-ordered labels; a contiguous ascending run is stored compactly as a range,
    // anything else stays explicit (order preserved).
    [[nodiscard]] static Units of(std::vector<std::int64_t> values);
    // Parse the grammar above; unrecognised / empty text is none.
    [[nodiscard]] static Units parse(std::string_view text);

    [[nodiscard]] bool empty() const noexcept; // true for none
    [[nodiscard]] std::int64_t size() const noexcept; // unit count (rows)
    [[nodiscard]] std::int64_t at(std::int64_t index) const; // unit at a row
    [[nodiscard]] std::vector<std::int64_t> to_vector() const; // materialise
    [[nodiscard]] std::string format() const; // "none" | "[...]"

    [[nodiscard]] bool operator==(const Units& other) const;
    [[nodiscard]] bool operator!=(const Units& other) const { return !(*this == other); }

private:
    // none: count_ == 0 and explicit_ empty.
    // range: count_ > 0 and explicit_ empty -> [start_, start_ + count_).
    // explicit: explicit_ non-empty (row i is explicit_[i]).
    std::int64_t start_ = 0;
    std::int64_t count_ = 0;
    std::vector<std::int64_t> explicit_;
};

} // namespace leakflow
