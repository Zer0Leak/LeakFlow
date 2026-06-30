#pragma once

#include <cstdint>
#include <vector>

namespace leakflow {

// Vector-clock buffer provenance (Phase 27). A clock is a dense per-element
// production-count vector; index 0 is reserved and value 0 means "this buffer is
// not downstream of element i" (a wildcard during matching). See
// docs/design/dataflow_sync_model.md Section 6.

// Two clocks are compatible iff every index that is non-zero in *both* holds the
// same value (zeros are wildcards). Shorter clocks are treated as zero-extended.
[[nodiscard]] bool provenance_compatible(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b);

// Fold-and-detect: component-wise max of the clocks, zero-extended to `size`.
// The first non-zero to claim an index sets the expected value; any later
// disagreeing non-zero is a conflict. Throws std::invalid_argument on conflict.
// Correct for any number of clocks (anchoring on one would miss conflicts among
// the others where the anchor is zero). Runs in O(total non-zero entries).
[[nodiscard]] std::vector<std::uint32_t> merge_provenance(
    const std::vector<const std::vector<std::uint32_t>*>& clocks,
    std::size_t size);

// A scalar "generation" for a buffer's clock: the maximum production count over
// all slots (0 for an empty clock). Used only for diagnostics/UI change
// detection (it replaces the old global Buffer::epoch stamp); it is not used for
// matching, which is the full fold (merge_provenance).
[[nodiscard]] std::uint32_t provenance_generation(const std::vector<std::uint32_t>& clock);

} // namespace leakflow
