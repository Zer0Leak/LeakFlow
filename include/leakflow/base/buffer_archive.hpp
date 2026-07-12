#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <torch/torch.h>

namespace leakflow::base {

// Storage-neutral sink/source for a Buffer's PAYLOAD body. A PayloadCodec writes
// its payload as named tensors and scalars through a BufferArchiveWriter and
// rebuilds it through a BufferArchiveReader, without knowing the concrete backend
// (HDF5 today, Zarr later). Torch-aware but storage-agnostic on purpose: it lives
// in leakflow_base so payload codecs in base and crypto can serialize through it
// without depending on the HDF5 backend in leakflow_extras. The Buffer envelope
// (caps + metadata) is handled by the file element, not by these interfaces.
// See docs/context/ARCHITECTURE_CONTRACTS.md (Buffer Persistence).
class BufferArchiveWriter {
public:
    virtual ~BufferArchiveWriter() = default;

    // Write one named tensor (materialized on CPU by the backend). Names are
    // codec-chosen and scoped to the payload.
    virtual void write_tensor(std::string_view name, const torch::Tensor& tensor) = 0;
    // Write a named scalar. Tensor and scalar name spaces are independent.
    virtual void write_int(std::string_view name, std::int64_t value) = 0;
    virtual void write_string(std::string_view name, std::string_view value) = 0;
};

class BufferArchiveReader {
public:
    virtual ~BufferArchiveReader() = default;

    // True when a payload entry (tensor or scalar) with this name exists. Codecs
    // use it to read optional entries and variable-length collections.
    [[nodiscard]] virtual bool has(std::string_view name) const = 0;
    [[nodiscard]] virtual torch::Tensor read_tensor(std::string_view name) const = 0;
    [[nodiscard]] virtual std::int64_t read_int(std::string_view name) const = 0;
    [[nodiscard]] virtual std::string read_string(std::string_view name) const = 0;
};

} // namespace leakflow::base
