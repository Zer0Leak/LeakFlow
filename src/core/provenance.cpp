#include "leakflow/core/provenance.hpp"

#include <cstddef>
#include <stdexcept>

namespace leakflow {

bool provenance_compatible(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b)
{
    const auto common = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < common; ++i) {
        if (a[i] != 0 && b[i] != 0 && a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

std::vector<std::uint32_t> merge_provenance(
    const std::vector<const std::vector<std::uint32_t>*>& clocks,
    std::size_t size)
{
    std::vector<std::uint32_t> merged(size, 0u);

    for (const auto* clock : clocks) {
        if (clock == nullptr) {
            continue;
        }
        const auto count = clock->size() < size ? clock->size() : size;
        for (std::size_t i = 0; i < count; ++i) {
            const auto value = (*clock)[i];
            if (value == 0) {
                continue; // wildcard
            }
            if (merged[i] == 0) {
                merged[i] = value; // first to claim index i
            } else if (merged[i] != value) {
                throw std::invalid_argument(
                    "provenance conflict: inputs are from incompatible buffer generations");
            }
        }
    }

    return merged;
}

std::uint32_t provenance_generation(const std::vector<std::uint32_t>& clock)
{
    std::uint32_t generation = 0;
    for (const auto value : clock) {
        if (value > generation) {
            generation = value;
        }
    }
    return generation;
}

} // namespace leakflow
