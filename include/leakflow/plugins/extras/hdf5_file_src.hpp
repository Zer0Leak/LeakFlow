#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <optional>
#include <string>

namespace leakflow::plugins::extras {

// Loads a LeakFlow tensor-dataset HDF5 file through chunked hyperslab reads,
// then emits one complete Torch payload per available semantic output pad.
class Hdf5FileSrc final : public Element {
public:
  explicit Hdf5FileSrc(std::string name = "hdf5filesrc0");

  [[nodiscard]] static ElementDescriptor descriptor();
  [[nodiscard]] std::optional<Buffer>
  process(std::optional<Buffer> input) override;
  [[nodiscard]] ElementOutputs process_pads(ElementInputs inputs) override;
};

} // namespace leakflow::plugins::extras
