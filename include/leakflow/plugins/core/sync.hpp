#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::core {

// Sync element (live phase, S11.4): the single front door for custom cross-source
// pairing. Takes N input pads (in_%u) and emits N aligned output pads (out_%u),
// pairing them by a tuned policy. It claims its own provenance slot and stamps
// every aligned output of a firing with the same value on that slot -- "injecting
// a common ancestor" -- so N independent input streams become common-origin
// downstream and every downstream join uses the plain default barrier.
//
// Offline (one buffer per pad) the policy is AllRequiredOnce: the executor gathers
// all connected inputs, Sync maps in_K -> out_K, and the executor's per-firing
// own-slot stamp does the common-ancestor injection. Streaming policies
// (Zip/Held/Latest/Barrier) become meaningful with the threaded pump.
class Sync final : public Element {
public:
    explicit Sync(std::string name = "sync0");

    [[nodiscard]] static ElementDescriptor descriptor();
    [[nodiscard]] std::optional<Buffer> process(std::optional<Buffer> input) override;
    [[nodiscard]] ElementOutputs process_pads(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::core
