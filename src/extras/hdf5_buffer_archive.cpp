#include "leakflow/extras/hdf5_buffer_archive.hpp"

#include <hdf5.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace leakflow::extras {
namespace {

inline constexpr auto schema_name = "leakflow.buffer";
inline constexpr auto schema_version = "1";
inline constexpr auto caps_group = "/caps";
inline constexpr auto metadata_group = "/metadata";
inline constexpr auto payload_group = "/payload";
inline constexpr auto caps_type_attr = "caps.type";
inline constexpr auto payload_type_attr = "payload.type";

// Minimal owning HDF5 handle; closes with the supplied closer on scope exit.
class Hdf5Id {
public:
    Hdf5Id() = default;
    Hdf5Id(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
    ~Hdf5Id() { reset(); }

    Hdf5Id(const Hdf5Id&) = delete;
    Hdf5Id& operator=(const Hdf5Id&) = delete;
    Hdf5Id(Hdf5Id&& other) noexcept
        : id_(std::exchange(other.id_, H5I_INVALID_HID)), close_(std::exchange(other.close_, nullptr)) {}
    Hdf5Id& operator=(Hdf5Id&& other) noexcept {
        if (this != &other) {
            reset();
            id_ = std::exchange(other.id_, H5I_INVALID_HID);
            close_ = std::exchange(other.close_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] hid_t get() const noexcept { return id_; }

private:
    void reset() noexcept {
        if (id_ >= 0 && close_ != nullptr) {
            (void)close_(id_);
        }
        id_ = H5I_INVALID_HID;
        close_ = nullptr;
    }

    hid_t id_ = H5I_INVALID_HID;
    herr_t (*close_)(hid_t) = nullptr;
};

[[noreturn]] void fail(std::string_view operation) {
    throw std::runtime_error("HDF5 buffer archive " + std::string(operation) + " failed");
}

hid_t checked(hid_t id, std::string_view operation) {
    if (id < 0) {
        fail(operation);
    }
    return id;
}

void check(herr_t status, std::string_view operation) {
    if (status < 0) {
        fail(operation);
    }
}

// --- string attributes (fixed-length, null-terminated) ---

void write_string_attribute(hid_t loc, const std::string& name, std::string_view value) {
    Hdf5Id space(checked(H5Screate(H5S_SCALAR), "attribute space"), H5Sclose);
    Hdf5Id type(checked(H5Tcopy(H5T_C_S1), "string type"), H5Tclose);
    check(H5Tset_size(type.get(), value.size() + 1), "string width");
    check(H5Tset_strpad(type.get(), H5T_STR_NULLTERM), "string padding");
    if (H5Aexists(loc, name.c_str()) > 0) {
        check(H5Adelete(loc, name.c_str()), "attribute replace");
    }
    Hdf5Id attribute(
        checked(H5Acreate2(loc, name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT),
            "attribute create"),
        H5Aclose);
    const std::string terminated(value);
    check(H5Awrite(attribute.get(), type.get(), terminated.c_str()), "attribute write");
}

[[nodiscard]] std::string read_string_attribute(hid_t attribute) {
    Hdf5Id type(checked(H5Aget_type(attribute), "attribute type"), H5Tclose);
    if (H5Tis_variable_str(type.get()) > 0) {
        char* value = nullptr;
        check(H5Aread(attribute, type.get(), &value), "variable string attribute read");
        std::string result = value == nullptr ? std::string() : std::string(value);
        if (value != nullptr) {
            (void)H5free_memory(value);
        }
        return result;
    }
    const auto width = H5Tget_size(type.get());
    std::vector<char> data(width + 1, '\0');
    check(H5Aread(attribute, type.get(), data.data()), "fixed string attribute read");
    return std::string(data.data());
}

[[nodiscard]] std::string read_string_attribute_by_name(hid_t loc, const std::string& name) {
    Hdf5Id attribute(checked(H5Aopen(loc, name.c_str(), H5P_DEFAULT), "attribute open"), H5Aclose);
    return read_string_attribute(attribute.get());
}

struct AttributeCollectState {
    std::map<std::string, std::string>* out = nullptr;
    std::string error;
};

herr_t collect_attribute(hid_t location, const char* name, const H5A_info_t*, void* opaque) noexcept {
    auto& state = *static_cast<AttributeCollectState*>(opaque);
    try {
        Hdf5Id attribute(H5Aopen(location, name, H5P_DEFAULT), H5Aclose);
        if (attribute.get() < 0) {
            state.error = "attribute open";
            return -1;
        }
        state.out->emplace(name, read_string_attribute(attribute.get()));
        return 0;
    } catch (const std::exception& error) {
        state.error = error.what();
        return -1;
    }
}

[[nodiscard]] std::map<std::string, std::string> read_group_attributes(hid_t file, const char* group_path) {
    std::map<std::string, std::string> attributes;
    if (H5Lexists(file, group_path, H5P_DEFAULT) <= 0) {
        return attributes;
    }
    Hdf5Id group(checked(H5Gopen2(file, group_path, H5P_DEFAULT), "group open"), H5Gclose);
    AttributeCollectState state{.out = &attributes};
    auto index = hsize_t{0};
    const auto status =
        H5Aiterate2(group.get(), H5_INDEX_NAME, H5_ITER_INC, &index, collect_attribute, &state);
    if (!state.error.empty()) {
        fail("attribute iteration");
    }
    check(status, "attribute iteration");
    return attributes;
}

// --- Torch <-> HDF5 dtype mapping (native strided CPU tensors) ---

[[nodiscard]] hid_t file_dtype(torch::ScalarType dtype) {
    switch (dtype) {
    case torch::kFloat32:
        return H5T_IEEE_F32LE;
    case torch::kFloat64:
        return H5T_IEEE_F64LE;
    case torch::kInt64:
        return H5T_STD_I64LE;
    case torch::kInt32:
        return H5T_STD_I32LE;
    case torch::kInt16:
        return H5T_STD_I16LE;
    case torch::kInt8:
        return H5T_STD_I8LE;
    case torch::kUInt8:
        return H5T_STD_U8LE;
    default:
        throw std::invalid_argument("HDF5 buffer archive: unsupported tensor dtype for storage");
    }
}

[[nodiscard]] hid_t native_dtype(torch::ScalarType dtype) {
    switch (dtype) {
    case torch::kFloat32:
        return H5T_NATIVE_FLOAT;
    case torch::kFloat64:
        return H5T_NATIVE_DOUBLE;
    case torch::kInt64:
        return H5T_NATIVE_INT64;
    case torch::kInt32:
        return H5T_NATIVE_INT32;
    case torch::kInt16:
        return H5T_NATIVE_INT16;
    case torch::kInt8:
        return H5T_NATIVE_INT8;
    case torch::kUInt8:
        return H5T_NATIVE_UINT8;
    default:
        throw std::invalid_argument("HDF5 buffer archive: unsupported tensor dtype for storage");
    }
}

[[nodiscard]] torch::ScalarType torch_dtype(hid_t type) {
    const auto type_class = H5Tget_class(type);
    const auto width = H5Tget_size(type);
    if (type_class == H5T_FLOAT) {
        if (width == 4) {
            return torch::kFloat32;
        }
        if (width == 8) {
            return torch::kFloat64;
        }
    } else if (type_class == H5T_INTEGER) {
        if (H5Tget_sign(type) == H5T_SGN_NONE) {
            if (width == 1) {
                return torch::kUInt8;
            }
        } else {
            switch (width) {
            case 8:
                return torch::kInt64;
            case 4:
                return torch::kInt32;
            case 2:
                return torch::kInt16;
            case 1:
                return torch::kInt8;
            default:
                break;
            }
        }
    }
    throw std::invalid_argument("HDF5 buffer archive: unsupported dataset dtype on load");
}

void write_tensor_dataset(hid_t group, const std::string& name, const torch::Tensor& tensor_in) {
    const auto tensor = tensor_in.to(torch::kCPU).contiguous();
    std::vector<hsize_t> dims;
    dims.reserve(static_cast<std::size_t>(tensor.dim()));
    for (const auto dimension : tensor.sizes()) {
        dims.push_back(static_cast<hsize_t>(dimension));
    }
    Hdf5Id space(
        checked(dims.empty() ? H5Screate(H5S_SCALAR)
                             : H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr),
            "dataset space"),
        H5Sclose);
    Hdf5Id dataset(
        checked(H5Dcreate2(group, name.c_str(), file_dtype(tensor.scalar_type()), space.get(),
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "dataset create"),
        H5Dclose);
    if (tensor.numel() > 0) {
        check(H5Dwrite(dataset.get(), native_dtype(tensor.scalar_type()), H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, tensor.const_data_ptr()),
            "dataset write");
    }
}

[[nodiscard]] torch::Tensor read_tensor_dataset(hid_t group, const std::string& name) {
    Hdf5Id dataset(checked(H5Dopen2(group, name.c_str(), H5P_DEFAULT), "dataset open"), H5Dclose);
    Hdf5Id type(checked(H5Dget_type(dataset.get()), "dataset type"), H5Tclose);
    const auto dtype = torch_dtype(type.get());
    Hdf5Id space(checked(H5Dget_space(dataset.get()), "dataset space"), H5Sclose);
    const auto rank = H5Sget_simple_extent_ndims(space.get());
    if (rank < 0) {
        fail("dataset rank");
    }
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
    if (rank > 0) {
        check(H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr) < 0 ? -1 : 0, "dataset dims");
    }
    std::vector<std::int64_t> shape(dims.begin(), dims.end());
    auto output = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    if (output.numel() > 0) {
        check(H5Dread(dataset.get(), native_dtype(dtype), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  output.mutable_data_ptr()),
            "dataset read");
    }
    return output;
}

} // namespace

// --- Writer ---

struct Hdf5BufferArchiveWriter::Impl {
    Hdf5Id file;
    Hdf5Id caps;
    Hdf5Id metadata;
    Hdf5Id payload;
};

Hdf5BufferArchiveWriter::Hdf5BufferArchiveWriter(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>()) {
    impl_->file = Hdf5Id(
        checked(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "file create"),
        H5Fclose);
    write_string_attribute(impl_->file.get(), "leakflow.schema", schema_name);
    write_string_attribute(impl_->file.get(), "leakflow.schema.version", schema_version);
    impl_->caps = Hdf5Id(
        checked(H5Gcreate2(impl_->file.get(), caps_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "caps group"),
        H5Gclose);
    impl_->metadata = Hdf5Id(
        checked(
            H5Gcreate2(impl_->file.get(), metadata_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "metadata group"),
        H5Gclose);
    impl_->payload = Hdf5Id(
        checked(
            H5Gcreate2(impl_->file.get(), payload_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "payload group"),
        H5Gclose);
}

Hdf5BufferArchiveWriter::~Hdf5BufferArchiveWriter() = default;
Hdf5BufferArchiveWriter::Hdf5BufferArchiveWriter(Hdf5BufferArchiveWriter&&) noexcept = default;
Hdf5BufferArchiveWriter& Hdf5BufferArchiveWriter::operator=(Hdf5BufferArchiveWriter&&) noexcept =
    default;

void Hdf5BufferArchiveWriter::set_caps_type(std::string_view value) {
    write_string_attribute(impl_->file.get(), caps_type_attr, value);
}

void Hdf5BufferArchiveWriter::set_payload_type(std::string_view value) {
    write_string_attribute(impl_->file.get(), payload_type_attr, value);
}

void Hdf5BufferArchiveWriter::set_caps_param(std::string_view key, std::string_view value) {
    write_string_attribute(impl_->caps.get(), std::string(key), value);
}

void Hdf5BufferArchiveWriter::set_metadata(std::string_view key, std::string_view value) {
    write_string_attribute(impl_->metadata.get(), std::string(key), value);
}

void Hdf5BufferArchiveWriter::write_tensor(std::string_view name, const torch::Tensor& tensor) {
    write_tensor_dataset(impl_->payload.get(), std::string(name), tensor);
}

void Hdf5BufferArchiveWriter::write_int(std::string_view name, std::int64_t value) {
    Hdf5Id space(checked(H5Screate(H5S_SCALAR), "int attribute space"), H5Sclose);
    const std::string attribute_name(name);
    if (H5Aexists(impl_->payload.get(), attribute_name.c_str()) > 0) {
        check(H5Adelete(impl_->payload.get(), attribute_name.c_str()), "int attribute replace");
    }
    Hdf5Id attribute(
        checked(H5Acreate2(impl_->payload.get(), attribute_name.c_str(), H5T_STD_I64LE, space.get(),
                    H5P_DEFAULT, H5P_DEFAULT),
            "int attribute create"),
        H5Aclose);
    check(H5Awrite(attribute.get(), H5T_NATIVE_INT64, &value), "int attribute write");
}

void Hdf5BufferArchiveWriter::write_string(std::string_view name, std::string_view value) {
    write_string_attribute(impl_->payload.get(), std::string(name), value);
}

// --- Reader ---

struct Hdf5BufferArchiveReader::Impl {
    Hdf5Id file;
    Hdf5Id payload;
    std::string caps_type;
    std::string payload_type;
    std::map<std::string, std::string> caps_params;
    std::map<std::string, std::string> metadata;
};

Hdf5BufferArchiveReader::Hdf5BufferArchiveReader(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>()) {
    impl_->file = Hdf5Id(
        checked(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "file open"), H5Fclose);
    if (H5Aexists(impl_->file.get(), caps_type_attr) <= 0
        || H5Aexists(impl_->file.get(), payload_type_attr) <= 0) {
        throw std::invalid_argument(
            "HDF5 buffer archive is missing caps.type or payload.type: " + path.string());
    }
    impl_->caps_type = read_string_attribute_by_name(impl_->file.get(), caps_type_attr);
    impl_->payload_type = read_string_attribute_by_name(impl_->file.get(), payload_type_attr);
    impl_->caps_params = read_group_attributes(impl_->file.get(), caps_group);
    impl_->metadata = read_group_attributes(impl_->file.get(), metadata_group);
    impl_->payload = Hdf5Id(
        checked(H5Gopen2(impl_->file.get(), payload_group, H5P_DEFAULT), "payload group open"),
        H5Gclose);
}

Hdf5BufferArchiveReader::~Hdf5BufferArchiveReader() = default;
Hdf5BufferArchiveReader::Hdf5BufferArchiveReader(Hdf5BufferArchiveReader&&) noexcept = default;
Hdf5BufferArchiveReader& Hdf5BufferArchiveReader::operator=(Hdf5BufferArchiveReader&&) noexcept =
    default;

const std::string& Hdf5BufferArchiveReader::caps_type() const { return impl_->caps_type; }
const std::string& Hdf5BufferArchiveReader::payload_type() const { return impl_->payload_type; }

const std::map<std::string, std::string>& Hdf5BufferArchiveReader::caps_params() const {
    return impl_->caps_params;
}

const std::map<std::string, std::string>& Hdf5BufferArchiveReader::metadata() const {
    return impl_->metadata;
}

bool Hdf5BufferArchiveReader::has(std::string_view name) const {
    const std::string entry(name);
    return H5Lexists(impl_->payload.get(), entry.c_str(), H5P_DEFAULT) > 0
        || H5Aexists(impl_->payload.get(), entry.c_str()) > 0;
}

torch::Tensor Hdf5BufferArchiveReader::read_tensor(std::string_view name) const {
    return read_tensor_dataset(impl_->payload.get(), std::string(name));
}

std::int64_t Hdf5BufferArchiveReader::read_int(std::string_view name) const {
    const std::string attribute_name(name);
    Hdf5Id attribute(
        checked(H5Aopen(impl_->payload.get(), attribute_name.c_str(), H5P_DEFAULT), "int attribute open"),
        H5Aclose);
    std::int64_t value = 0;
    check(H5Aread(attribute.get(), H5T_NATIVE_INT64, &value), "int attribute read");
    return value;
}

std::string Hdf5BufferArchiveReader::read_string(std::string_view name) const {
    return read_string_attribute_by_name(impl_->payload.get(), std::string(name));
}

} // namespace leakflow::extras
