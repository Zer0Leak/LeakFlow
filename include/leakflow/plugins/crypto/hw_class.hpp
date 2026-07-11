#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::crypto {

// Combines two Hamming-weight leakage channels into a single true class label per trace
// (Convert/SCA/HwClass). Consumes an AesLeakage tensor `[B, N, C]` with the first two channels
// being HW(m) and HW(y), and emits int64 class labels `[B, N] = (hw_max+1)*HW(m) + HW(y)` --
// the 81-class (for AES, hw_max=8) ground truth that ClusteringStats scores a GMM against.
class HwClass final : public Element {
public:
    explicit HwClass(std::string name = "hwclass0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
};

} // namespace leakflow::plugins::crypto
