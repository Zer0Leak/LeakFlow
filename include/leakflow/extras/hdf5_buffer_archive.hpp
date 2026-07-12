#pragma once

#include "leakflow/base/buffer_archive.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace leakflow::extras {

// HDF5 backend for whole-Buffer persistence (the `leakflow.buffer` schema),
// following the conventions of the trace tensor-dataset files: native datasets
// for tensors and attributes for the envelope. Layout:
//
//   / (root)   attrs leakflow.schema="leakflow.buffer", leakflow.schema.version,
//                    caps.type, payload.type
//   /caps      attrs: one per caps parameter
//   /metadata  attrs: the Buffer's metadata map
//   /payload   datasets: the codec's named tensors; attrs: the codec's scalars
//
// HDF5 is fully hidden behind a pimpl so this header stays HDF5-free (leakflow_extras
// links HDF5 privately). The envelope setters/getters are used by the file element;
// the write_*/read_* payload methods implement the base archive interface for codecs.
class Hdf5BufferArchiveWriter final : public leakflow::base::BufferArchiveWriter {
public:
    explicit Hdf5BufferArchiveWriter(const std::filesystem::path& path);
    ~Hdf5BufferArchiveWriter() override;
    Hdf5BufferArchiveWriter(Hdf5BufferArchiveWriter&&) noexcept;
    Hdf5BufferArchiveWriter& operator=(Hdf5BufferArchiveWriter&&) noexcept;

    // Envelope (element-level).
    void set_caps_type(std::string_view value);
    void set_payload_type(std::string_view value);
    void set_caps_param(std::string_view key, std::string_view value);
    void set_metadata(std::string_view key, std::string_view value);

    // Payload body (codec-level, base interface).
    void write_tensor(std::string_view name, const torch::Tensor& tensor) override;
    void write_int(std::string_view name, std::int64_t value) override;
    void write_string(std::string_view name, std::string_view value) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Hdf5BufferArchiveReader final : public leakflow::base::BufferArchiveReader {
public:
    explicit Hdf5BufferArchiveReader(const std::filesystem::path& path);
    ~Hdf5BufferArchiveReader() override;
    Hdf5BufferArchiveReader(Hdf5BufferArchiveReader&&) noexcept;
    Hdf5BufferArchiveReader& operator=(Hdf5BufferArchiveReader&&) noexcept;

    // Envelope (element-level).
    [[nodiscard]] const std::string& caps_type() const;
    [[nodiscard]] const std::string& payload_type() const;
    [[nodiscard]] const std::map<std::string, std::string>& caps_params() const;
    [[nodiscard]] const std::map<std::string, std::string>& metadata() const;

    // Payload body (codec-level, base interface).
    [[nodiscard]] bool has(std::string_view name) const override;
    [[nodiscard]] torch::Tensor read_tensor(std::string_view name) const override;
    [[nodiscard]] std::int64_t read_int(std::string_view name) const override;
    [[nodiscard]] std::string read_string(std::string_view name) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace leakflow::extras
