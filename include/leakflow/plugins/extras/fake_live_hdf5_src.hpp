#pragma once

#include "leakflow/core/buffer.hpp"
#include "leakflow/core/element.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace leakflow::plugins::extras {

// Replays aligned row batches from a LeakFlow tensor-dataset HDF5 file as a
// deterministic finite live stream.
class FakeLiveHdf5Src final : public Element {
public:
  explicit FakeLiveHdf5Src(std::string name = "fakelivehdf5src0");
  ~FakeLiveHdf5Src() override;

  FakeLiveHdf5Src(const FakeLiveHdf5Src &) = delete;
  FakeLiveHdf5Src &operator=(const FakeLiveHdf5Src &) = delete;

  [[nodiscard]] static ElementDescriptor descriptor();
  void start() override;
  [[nodiscard]] std::optional<Buffer>
  process(std::optional<Buffer> input) override;
  [[nodiscard]] ElementOutputs process_pads(ElementInputs inputs) override;
  [[nodiscard]] bool at_end_of_stream() const override;
  [[nodiscard]] bool can_replay() const override;
  void stop() override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace leakflow::plugins::extras
