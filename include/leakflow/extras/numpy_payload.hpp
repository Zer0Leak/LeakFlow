#pragma once

#include "leakflow/core/caps.hpp"
#include "leakflow/core/payload.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <cnpy++.hpp>

namespace leakflow::extras {

inline constexpr auto numpy_array_caps_type = "leakflow/numpy-array";

class NumpyPayload final : public Payload {
public:
    explicit NumpyPayload(cnpypp::NpyArray array);
    NumpyPayload(cnpypp::NpyArray array, char dtype_code);

    [[nodiscard]] std::string type_name() const override;
    [[nodiscard]] std::string layout() const override;
    void describe(SummarySection& section, std::int64_t summary_level) const override;

    [[nodiscard]] const cnpypp::NpyArray& array() const;
    [[nodiscard]] cnpypp::NpyArray& array();

    [[nodiscard]] const std::vector<std::uint64_t>& shape() const;
    [[nodiscard]] cnpypp::MemoryOrder memory_order() const;
    [[nodiscard]] std::uint64_t rank() const;
    [[nodiscard]] std::uint64_t element_count() const;
    [[nodiscard]] std::uint64_t byte_count() const;
    [[nodiscard]] unsigned word_size() const;
    [[nodiscard]] char dtype_code() const;
    [[nodiscard]] std::string dtype_name() const;
    [[nodiscard]] std::string device_name() const;
    [[nodiscard]] Caps caps() const;

private:
    cnpypp::NpyArray array_;
    char dtype_code_ = '\0';
};

[[nodiscard]] NumpyPayload load_npy(const std::filesystem::path& path);

} // namespace leakflow::extras
