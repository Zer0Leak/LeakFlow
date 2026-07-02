#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <torch/torch.h>

namespace leakflow::plugins::base {

// Simulated live-capture source (live phase). Loads a Torch `.pt` tensor and
// streams ONE Buffer per entry along axis 0 (e.g. [50, 5000] -> 50 buffers of
// [1, 5000]), then signals end-of-stream. Declares itself live, so run() pumps
// until the stream ends. Deterministic; the driver for live tests.
// See docs/design/dataflow_sync_model.md S12 and docs/design/dataflow_sync_walkthroughs.md Ex.4.
class FakeLiveSrc final : public Element {
public:
    explicit FakeLiveSrc(std::string name = "fakelivesrc0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] bool at_end_of_stream() const override;
    void stop() override;

private:
    void ensure_loaded();

    torch::Tensor tensor_;
    std::int64_t cursor_ = 0;
    bool loaded_ = false;
};

} // namespace leakflow::plugins::base
