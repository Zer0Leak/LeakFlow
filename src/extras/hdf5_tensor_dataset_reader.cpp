#include "leakflow/extras/hdf5_tensor_dataset_reader.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace leakflow::extras {
namespace {

class Hdf5Handle {
public:
  Hdf5Handle() = default;

  Hdf5Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}

  ~Hdf5Handle() { reset(); }

  Hdf5Handle(const Hdf5Handle &) = delete;
  Hdf5Handle &operator=(const Hdf5Handle &) = delete;

  Hdf5Handle(Hdf5Handle &&other) noexcept
      : id_(std::exchange(other.id_, H5I_INVALID_HID)),
        close_(std::exchange(other.close_, nullptr)) {}

  Hdf5Handle &operator=(Hdf5Handle &&other) noexcept {
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

[[noreturn]] void throw_hdf5_error(std::string_view operation,
                                   std::string_view path = {}) {
  auto message = std::string("HDF5 ") + std::string(operation) + " failed";
  if (!path.empty()) {
    message += " for " + std::string(path);
  }
  throw std::runtime_error(std::move(message));
}

[[nodiscard]] Hdf5Handle checked_handle(hid_t id, herr_t (*close)(hid_t),
                                        std::string_view operation,
                                        std::string_view path = {}) {
  if (id < 0) {
    throw_hdf5_error(operation, path);
  }
  return Hdf5Handle(id, close);
}

void check_hdf5(herr_t status, std::string_view operation,
                std::string_view path = {}) {
  if (status < 0) {
    throw_hdf5_error(operation, path);
  }
}

[[nodiscard]] std::string format_float(double value) {
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return stream.str();
}

template <typename Value, typename Formatter>
[[nodiscard]] std::string format_values(const std::vector<Value> &values,
                                        Formatter formatter) {
  if (values.size() == 1) {
    return formatter(values.front());
  }

  std::string result = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      result += ',';
    }
    result += formatter(values[index]);
  }
  result += ']';
  return result;
}

[[nodiscard]] std::size_t attribute_value_count(hid_t attribute) {
  const auto space = checked_handle(H5Aget_space(attribute), H5Sclose,
                                    "attribute dataspace inspection");
  const auto points = H5Sget_simple_extent_npoints(space.get());
  if (points < 0) {
    throw_hdf5_error("attribute extent inspection");
  }
  if (static_cast<std::uint64_t>(points) >
      std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("HDF5 attribute is too large to inspect");
  }
  return static_cast<std::size_t>(points);
}

[[nodiscard]] std::string read_string_attribute(hid_t attribute, hid_t type,
                                                std::size_t count) {
  if (count == 0) {
    return "[]";
  }

  if (H5Tis_variable_str(type) > 0) {
    if (count != 1) {
      throw std::invalid_argument(
          "variable-length HDF5 string arrays are not supported as metadata");
    }
    char *value = nullptr;
    check_hdf5(H5Aread(attribute, type, &value),
               "variable string attribute read");
    const auto result = value == nullptr ? std::string() : std::string(value);
    if (value != nullptr) {
      check_hdf5(H5free_memory(value), "variable string attribute release");
    }
    return result;
  }

  const auto width = H5Tget_size(type);
  if (width == 0 ||
      (count != 0 && width > std::numeric_limits<std::size_t>::max() / count)) {
    throw std::overflow_error(
        "fixed HDF5 string attribute is too large to inspect");
  }

  std::vector<char> data(width * count, '\0');
  check_hdf5(H5Aread(attribute, type, data.data()),
             "fixed string attribute read");
  std::vector<std::string> values;
  values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto *begin = data.data() + index * width;
    auto length = std::size_t{0};
    while (length < width && begin[length] != '\0') {
      ++length;
    }
    while (length > 0 && begin[length - 1] == ' ') {
      --length;
    }
    values.emplace_back(begin, length);
  }
  return format_values(values, [](const std::string &value) { return value; });
}

[[nodiscard]] std::string read_enum_attribute(hid_t attribute, hid_t type,
                                              std::size_t count) {
  if (count != 1) {
    throw std::invalid_argument("HDF5 enum metadata must be scalar");
  }

  const auto native_type =
      checked_handle(H5Tget_native_type(type, H5T_DIR_ASCEND), H5Tclose,
                     "enum native type inspection");
  const auto width = H5Tget_size(native_type.get());
  if (width == 0) {
    throw_hdf5_error("enum width inspection");
  }
  std::vector<unsigned char> data(width, 0);
  check_hdf5(H5Aread(attribute, native_type.get(), data.data()),
             "enum attribute read");

  std::vector<char> name(256, '\0');
  check_hdf5(
      H5Tenum_nameof(native_type.get(), data.data(), name.data(), name.size()),
      "enum attribute name lookup");
  auto result = std::string(name.data());
  if (result == "TRUE") {
    return "true";
  }
  if (result == "FALSE") {
    return "false";
  }
  return result;
}

[[nodiscard]] std::string read_attribute_value(hid_t attribute) {
  const auto type = checked_handle(H5Aget_type(attribute), H5Tclose,
                                   "attribute type inspection");
  const auto count = attribute_value_count(attribute);
  const auto type_class = H5Tget_class(type.get());

  switch (type_class) {
  case H5T_STRING:
    return read_string_attribute(attribute, type.get(), count);
  case H5T_ENUM:
    return read_enum_attribute(attribute, type.get(), count);
  case H5T_INTEGER: {
    if (H5Tget_sign(type.get()) == H5T_SGN_NONE) {
      std::vector<unsigned long long> values(count);
      check_hdf5(H5Aread(attribute, H5T_NATIVE_ULLONG, values.data()),
                 "unsigned attribute read");
      return format_values(values,
                           [](auto value) { return std::to_string(value); });
    }
    std::vector<long long> values(count);
    check_hdf5(H5Aread(attribute, H5T_NATIVE_LLONG, values.data()),
               "signed attribute read");
    return format_values(values,
                         [](auto value) { return std::to_string(value); });
  }
  case H5T_FLOAT: {
    std::vector<double> values(count);
    check_hdf5(H5Aread(attribute, H5T_NATIVE_DOUBLE, values.data()),
               "floating attribute read");
    return format_values(values,
                         [](auto value) { return format_float(value); });
  }
  default:
    throw std::invalid_argument("unsupported HDF5 metadata attribute type");
  }
}

struct AttributeIterationState {
  TensorDatasetAttributes *attributes = nullptr;
  std::exception_ptr error;
};

herr_t collect_attribute(hid_t object, const char *name, const H5A_info_t *,
                         void *opaque_state) noexcept {
  auto &state = *static_cast<AttributeIterationState *>(opaque_state);
  try {
    const auto attribute = checked_handle(H5Aopen(object, name, H5P_DEFAULT),
                                          H5Aclose, "attribute open", name);
    state.attributes->emplace(name, read_attribute_value(attribute.get()));
    return 0;
  } catch (...) {
    state.error = std::current_exception();
    return -1;
  }
}

[[nodiscard]] TensorDatasetAttributes inspect_attributes(hid_t object) {
  TensorDatasetAttributes attributes;
  AttributeIterationState state{.attributes = &attributes};
  auto index = hsize_t{0};
  const auto status = H5Aiterate2(object, H5_INDEX_NAME, H5_ITER_INC, &index,
                                  collect_attribute, &state);
  if (state.error) {
    std::rethrow_exception(state.error);
  }
  check_hdf5(status, "attribute iteration");
  return attributes;
}

[[nodiscard]] TensorDatasetDType inspect_dataset_dtype(hid_t dataset,
                                                       std::string_view path) {
  const auto type = checked_handle(H5Dget_type(dataset), H5Tclose,
                                   "dataset type inspection", path);
  const auto type_class = H5Tget_class(type.get());
  const auto width = H5Tget_size(type.get());

  if (type_class == H5T_INTEGER && width == sizeof(std::uint8_t) &&
      H5Tget_sign(type.get()) == H5T_SGN_NONE) {
    return TensorDatasetDType::UInt8;
  }
  if (type_class == H5T_FLOAT && width == sizeof(float) &&
      H5Tget_precision(type.get()) == 32) {
    return TensorDatasetDType::Float32;
  }

  throw std::invalid_argument(
      "HDF5 tensor array " + std::string(path) +
      " has unsupported dtype; LeakFlow currently supports uint8 and float32");
}

[[nodiscard]] bool
inspect_row_alignment(const TensorDatasetAttributes &attributes,
                      const std::vector<std::int64_t> &shape,
                      std::string_view path) {
  const auto attribute = attributes.find("leakflow.row_aligned");
  if (attribute == attributes.end()) {
    return shape.size() > 1;
  }
  if (attribute->second == "true" || attribute->second == "1") {
    return true;
  }
  if (attribute->second == "false" || attribute->second == "0") {
    return false;
  }
  throw std::invalid_argument(
      "HDF5 tensor array " + std::string(path) +
      " has invalid leakflow.row_aligned metadata; expected true or false");
}

[[nodiscard]] std::vector<std::int64_t>
inspect_dataset_shape(hid_t dataset, std::string_view path) {
  const auto space = checked_handle(H5Dget_space(dataset), H5Sclose,
                                    "dataset dataspace inspection", path);
  const auto rank = H5Sget_simple_extent_ndims(space.get());
  if (rank <= 0) {
    throw std::invalid_argument("HDF5 tensor array " + std::string(path) +
                                " must have rank one or greater");
  }

  std::vector<hsize_t> dimensions(static_cast<std::size_t>(rank));
  if (H5Sget_simple_extent_dims(space.get(), dimensions.data(), nullptr) < 0) {
    throw_hdf5_error("dataset shape inspection", path);
  }

  std::vector<std::int64_t> shape;
  shape.reserve(dimensions.size());
  for (const auto dimension : dimensions) {
    if (dimension >
        static_cast<hsize_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::overflow_error(
          "HDF5 tensor array dimension exceeds the Torch shape range");
    }
    shape.push_back(static_cast<std::int64_t>(dimension));
  }
  return shape;
}

[[nodiscard]] std::string child_path(std::string_view parent,
                                     std::string_view child) {
  if (parent == "/") {
    return "/" + std::string(child);
  }
  return std::string(parent) + "/" + std::string(child);
}

[[nodiscard]] H5L_type_t inspect_link_type(hid_t group, const char *name,
                                           std::string_view path) {
#if H5_VERSION_GE(1, 12, 0)
  H5L_info2_t info{};
  check_hdf5(H5Lget_info2(group, name, &info, H5P_DEFAULT), "link inspection",
             path);
#else
  H5L_info1_t info{};
  check_hdf5(H5Lget_info1(group, name, &info, H5P_DEFAULT), "link inspection",
             path);
#endif
  return info.type;
}

struct GroupObjectIdentity {
  unsigned long file_number = 0;
#if H5_VERSION_GE(1, 12, 0)
  H5O_token_t token{};
#else
  haddr_t address = HADDR_UNDEF;
#endif
  std::string first_path;
};

struct GroupTraversalState {
  hid_t file = H5I_INVALID_HID;
  std::vector<GroupObjectIdentity> visited_groups;
};

[[nodiscard]] GroupObjectIdentity
inspect_group_identity(hid_t group, std::string_view path) {
  GroupObjectIdentity identity;
#if H5_VERSION_GE(1, 12, 0)
  H5O_info2_t info{};
  check_hdf5(H5Oget_info3(group, &info, H5O_INFO_BASIC),
             "group identity inspection", path);
  identity.file_number = info.fileno;
  identity.token = info.token;
#else
  H5O_info1_t info{};
  check_hdf5(H5Oget_info1(group, &info), "group identity inspection", path);
  identity.file_number = info.fileno;
  identity.address = info.addr;
#endif
  identity.first_path = path;
  return identity;
}

[[nodiscard]] bool same_group_object(hid_t file,
                                     const GroupObjectIdentity &left,
                                     const GroupObjectIdentity &right) {
  if (left.file_number != right.file_number) {
    return false;
  }
#if H5_VERSION_GE(1, 12, 0)
  auto comparison = 0;
  check_hdf5(H5Otoken_cmp(file, &left.token, &right.token, &comparison),
             "group identity comparison");
  return comparison == 0;
#else
  return left.address == right.address;
#endif
}

void register_group_object(hid_t group, std::string_view path,
                           GroupTraversalState &state) {
  auto identity = inspect_group_identity(group, path);
  const auto existing = std::ranges::find_if(
      state.visited_groups, [&state, &identity](const auto &visited) {
        return same_group_object(state.file, identity, visited);
      });
  if (existing != state.visited_groups.end()) {
    throw std::invalid_argument(
        "HDF5 hard-linked group alias or cycle is not supported: " +
        std::string(path) + " refers to the group already inspected at " +
        existing->first_path);
  }
  state.visited_groups.push_back(std::move(identity));
}

void inspect_group(hid_t file, std::string_view path,
                   TensorDatasetDescriptor &descriptor,
                   GroupTraversalState &traversal) {
  const auto group =
      checked_handle(H5Gopen2(file, std::string(path).c_str(), H5P_DEFAULT),
                     H5Gclose, "group open", path);
  register_group_object(group.get(), path, traversal);
  descriptor.groups.push_back({.path = std::string(path),
                               .attributes = inspect_attributes(group.get())});

  H5G_info_t group_info{};
  check_hdf5(H5Gget_info(group.get(), &group_info), "group inspection", path);
  for (hsize_t index = 0; index < group_info.nlinks; ++index) {
    const auto name_length =
        H5Lget_name_by_idx(group.get(), ".", H5_INDEX_NAME, H5_ITER_INC, index,
                           nullptr, 0, H5P_DEFAULT);
    if (name_length < 0) {
      throw_hdf5_error("group child-name inspection", path);
    }

    std::vector<char> name(static_cast<std::size_t>(name_length) + 1, '\0');
    if (H5Lget_name_by_idx(group.get(), ".", H5_INDEX_NAME, H5_ITER_INC, index,
                           name.data(), name.size(), H5P_DEFAULT) < 0) {
      throw_hdf5_error("group child-name read", path);
    }

    const auto object_path = child_path(path, name.data());
    const auto link_type =
        inspect_link_type(group.get(), name.data(), object_path);
    if (link_type != H5L_TYPE_HARD) {
      throw std::invalid_argument(
          "HDF5 tensor datasets support only hard links; unsupported link at " +
          object_path);
    }
    const auto object =
        checked_handle(H5Oopen(file, object_path.c_str(), H5P_DEFAULT),
                       H5Oclose, "object open", object_path);
    switch (H5Iget_type(object.get())) {
    case H5I_GROUP:
      inspect_group(file, object_path, descriptor, traversal);
      break;
    case H5I_DATASET: {
      const auto dataset =
          checked_handle(H5Dopen2(file, object_path.c_str(), H5P_DEFAULT),
                         H5Dclose, "dataset open", object_path);
      auto shape = inspect_dataset_shape(dataset.get(), object_path);
      auto attributes = inspect_attributes(dataset.get());
      descriptor.arrays.push_back({
          .path = object_path,
          .dtype = inspect_dataset_dtype(dataset.get(), object_path),
          .shape = shape,
          .row_aligned = inspect_row_alignment(attributes, shape, object_path),
          .attributes = std::move(attributes),
      });
      break;
    }
    default:
      break;
    }
  }
}

[[nodiscard]] torch::ScalarType torch_dtype(TensorDatasetDType dtype) {
  switch (dtype) {
  case TensorDatasetDType::UInt8:
    return torch::kUInt8;
  case TensorDatasetDType::Float32:
    return torch::kFloat32;
  }
  throw std::invalid_argument("unsupported tensor dataset dtype");
}

[[nodiscard]] hid_t hdf5_memory_dtype(TensorDatasetDType dtype) {
  switch (dtype) {
  case TensorDatasetDType::UInt8:
    return H5T_NATIVE_UINT8;
  case TensorDatasetDType::Float32:
    return H5T_NATIVE_FLOAT;
  }
  throw std::invalid_argument("unsupported tensor dataset dtype");
}

[[nodiscard]] std::uint64_t
row_byte_count(const TensorArrayDescriptor &descriptor) {
  if (!descriptor.row_aligned) {
    return descriptor.logical_bytes();
  }
  TensorArrayDescriptor one_row = descriptor;
  one_row.shape.front() = 1;
  return one_row.logical_bytes();
}

[[nodiscard]] std::vector<hsize_t>
to_hdf5_shape(const std::vector<std::int64_t> &shape) {
  std::vector<hsize_t> result;
  result.reserve(shape.size());
  for (const auto dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument("negative tensor dataset shape dimension");
    }
    result.push_back(static_cast<hsize_t>(dimension));
  }
  return result;
}

} // namespace

struct Hdf5TensorDatasetReader::Impl {
  explicit Impl(std::filesystem::path input_path)
      : file(checked_handle(
            H5Fopen(input_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose,
            "file open", input_path.string())) {
    dataset.storage_format = "hdf5";
    dataset.path = std::move(input_path);
    GroupTraversalState traversal{.file = file.get()};
    inspect_group(file.get(), "/", dataset, traversal);
    std::ranges::sort(dataset.groups, {}, &TensorGroupDescriptor::path);
    std::ranges::sort(dataset.arrays, {}, &TensorArrayDescriptor::path);
  }

  Hdf5Handle file;
  TensorDatasetDescriptor dataset;
};

Hdf5TensorDatasetReader::Hdf5TensorDatasetReader(std::filesystem::path path)
    : impl_(std::make_unique<Impl>(std::move(path))) {}

Hdf5TensorDatasetReader::~Hdf5TensorDatasetReader() = default;
Hdf5TensorDatasetReader::Hdf5TensorDatasetReader(
    Hdf5TensorDatasetReader &&) noexcept = default;
Hdf5TensorDatasetReader &Hdf5TensorDatasetReader::operator=(
    Hdf5TensorDatasetReader &&) noexcept = default;

const TensorDatasetDescriptor &
Hdf5TensorDatasetReader::descriptor() const noexcept {
  return impl_->dataset;
}

torch::Tensor Hdf5TensorDatasetReader::read_tensor(
    std::string_view requested_path, const TensorReadOptions &options,
    TensorReadProgressCallback progress) const {
  if (options.io_batch_rows == 0) {
    throw std::invalid_argument(
        "HDF5 tensor reads require io_batch_rows greater than zero");
  }
  if (requested_path.empty() || requested_path.front() != '/') {
    throw std::invalid_argument("HDF5 tensor array paths must be absolute");
  }

  // Report progress and honor cooperative cancellation: a false return aborts
  // the read with TensorReadCancelled rather than returning a partial tensor.
  const auto report = [&progress](const TensorReadProgress &update) {
    if (progress && !progress(update)) {
      throw TensorReadCancelled();
    }
  };

  const auto *array = impl_->dataset.find_array(requested_path);
  if (array == nullptr) {
    throw std::out_of_range("HDF5 tensor array does not exist: " +
                            std::string(requested_path));
  }

  const auto output_shape = array->selected_shape(options.rows);
  auto output = torch::empty(output_shape, torch::TensorOptions()
                                               .dtype(torch_dtype(array->dtype))
                                               .device(torch::kCPU));
  const auto total_bytes = array->selected_logical_bytes(options.rows);
  const auto dataset = checked_handle(
      H5Dopen2(impl_->file.get(), array->path.c_str(), H5P_DEFAULT), H5Dclose,
      "dataset open", array->path);

  if (!array->row_aligned) {
    if (total_bytes != 0) {
      check_hdf5(H5Dread(dataset.get(), hdf5_memory_dtype(array->dtype),
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, output.data_ptr()),
                 "fixed tensor read", array->path);
    }
    report({
        .array_path = array->path,
        .logical_bytes_read = total_bytes,
        .total_logical_bytes = total_bytes,
        .rows_read = 1,
        .total_rows = 1,
    });
    return output;
  }

  const auto selected_rows = static_cast<std::uint64_t>(output_shape.front());
  if (selected_rows == 0) {
    report({
        .array_path = array->path,
        .logical_bytes_read = 0,
        .total_logical_bytes = 0,
        .rows_read = 0,
        .total_rows = 0,
    });
    return output;
  }

  const auto file_space =
      checked_handle(H5Dget_space(dataset.get()), H5Sclose,
                     "dataset dataspace inspection", array->path);
  const auto bytes_per_row = row_byte_count(*array);
  auto rows_read = std::uint64_t{0};
  while (rows_read < selected_rows) {
    const auto chunk_rows =
        std::min(options.io_batch_rows, selected_rows - rows_read);
    auto chunk_shape = to_hdf5_shape(array->shape);
    chunk_shape.front() = static_cast<hsize_t>(chunk_rows);
    std::vector<hsize_t> offset(chunk_shape.size(), 0);
    offset.front() = static_cast<hsize_t>(options.rows.start + rows_read);

    check_hdf5(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET,
                                   offset.data(), nullptr, chunk_shape.data(),
                                   nullptr),
               "dataset hyperslab selection", array->path);
    const auto memory_space =
        checked_handle(H5Screate_simple(static_cast<int>(chunk_shape.size()),
                                        chunk_shape.data(), nullptr),
                       H5Sclose, "memory dataspace creation", array->path);

    auto *destination =
        static_cast<std::byte *>(output.data_ptr()) + rows_read * bytes_per_row;
    check_hdf5(H5Dread(dataset.get(), hdf5_memory_dtype(array->dtype),
                       memory_space.get(), file_space.get(), H5P_DEFAULT,
                       destination),
               "tensor hyperslab read", array->path);

    rows_read += chunk_rows;
    // Fires once per hyperslab; a false return here throws TensorReadCancelled
    // and unwinds the loop, discarding the partially-filled output tensor.
    report({
        .array_path = array->path,
        .logical_bytes_read = rows_read * bytes_per_row,
        .total_logical_bytes = total_bytes,
        .rows_read = rows_read,
        .total_rows = selected_rows,
    });
  }
  return output;
}

} // namespace leakflow::extras
