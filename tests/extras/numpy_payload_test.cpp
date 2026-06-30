#include "leakflow/core/buffer.hpp"
#include "leakflow/base/numeric_caps.hpp"
#include "leakflow/extras/numpy_payload.hpp"

#include <cnpy++.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }

    return true;
}

template <typename Exception, typename Function>
bool throws_exception(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    }

    return false;
}

std::filesystem::path temp_path(const char* filename)
{
    return std::filesystem::temp_directory_path() / filename;
}

} // namespace

int main()
{
    const auto float_path = temp_path("leakflow_numpy_payload_float.npy");
    const auto uint8_path = temp_path("leakflow_numpy_payload_uint8.npy");
    const auto fortran_path = temp_path("leakflow_numpy_payload_fortran.npy");

    const std::vector<float> floats{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    cnpypp::npy_save(float_path.string(), floats.begin(), {2, 3});

    auto payload = leakflow::extras::load_npy(float_path);
    if (!expect(payload.type_name() == "leakflow/numpy-array", "NumPy payload type name changed")) {
        return 1;
    }
    if (!expect(payload.rank() == 2, "NumPy payload rank mismatch")) {
        return 1;
    }
    if (!expect(payload.shape() == std::vector<std::uint64_t>{2, 3}, "NumPy payload shape mismatch")) {
        return 1;
    }
    if (!expect(payload.memory_order() == cnpypp::MemoryOrder::C, "NumPy payload memory order mismatch")) {
        return 1;
    }
    if (!expect(payload.element_count() == 6, "NumPy payload element count mismatch")) {
        return 1;
    }
    if (!expect(payload.word_size() == sizeof(float), "NumPy payload word size mismatch")) {
        return 1;
    }
    if (!expect(payload.dtype_code() == 'f', "NumPy payload dtype code mismatch")) {
        return 1;
    }
    if (!expect(payload.dtype_name() == "float32", "NumPy payload dtype name mismatch")) {
        return 1;
    }
    if (!expect(payload.device_name() == "cpu", "NumPy payload device name mismatch")) {
        return 1;
    }
    if (!expect(payload.byte_count() == sizeof(float) * floats.size(), "NumPy payload byte count mismatch")) {
        return 1;
    }
    if (!expect(payload.array().data<float>()[4] == 5.0F, "NumPy payload data pointer mismatch")) {
        return 1;
    }

    const auto payload_caps = payload.caps();
    if (!expect(payload_caps.type() == leakflow::extras::numpy_array_caps_type,
            "NumPy payload caps type mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_dtype) == "float32",
            "NumPy payload caps dtype mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_device) == "cpu",
            "NumPy payload caps device mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_rank) == "2",
            "NumPy payload caps rank mismatch")) {
        return 1;
    }
    if (!expect(payload_caps.param(leakflow::base::caps_param_shape) == "[2,3]",
            "NumPy payload caps shape mismatch")) {
        return 1;
    }

    leakflow::SummarySection summary_section("Payload");
    payload.describe(summary_section, 2);
    if (!expect(summary_section.fields.size() >= 8, "NumPy summary did not include detailed fields")) {
        return 1;
    }
    if (!expect(summary_section.fields[1].label == "dtype", "NumPy summary did not include dtype")) {
        return 1;
    }
    if (!expect(summary_section.fields[3].label == "shape", "NumPy summary did not include shape")) {
        return 1;
    }
    if (!expect(summary_section.fields[7].label == "memory_order", "NumPy summary did not include memory order")) {
        return 1;
    }

    leakflow::Buffer buffer(leakflow::Caps("leakflow/numpy-array"));
    buffer.set_payload(std::make_shared<leakflow::extras::NumpyPayload>(std::move(payload)));
    const auto roundtrip = buffer.payload_as<leakflow::extras::NumpyPayload>();
    if (!expect(roundtrip != nullptr, "buffer did not preserve NumpyPayload type")) {
        return 1;
    }
    if (!expect(roundtrip->shape() == std::vector<std::uint64_t>{2, 3}, "buffer NumPy payload shape mismatch")) {
        return 1;
    }

    const std::vector<std::uint8_t> bytes{1, 2, 3, 4};
    cnpypp::npy_save(uint8_path.string(), bytes.begin(), {4});
    auto byte_payload = leakflow::extras::load_npy(uint8_path);
    if (!expect(byte_payload.rank() == 1, "uint8 NumPy payload rank mismatch")) {
        return 1;
    }
    if (!expect(byte_payload.word_size() == sizeof(std::uint8_t), "uint8 NumPy payload word size mismatch")) {
        return 1;
    }
    if (!expect(byte_payload.dtype_name() == "uint8", "uint8 NumPy payload dtype name mismatch")) {
        return 1;
    }
    if (!expect(byte_payload.array().data<std::uint8_t>()[2] == 3, "uint8 NumPy payload data mismatch")) {
        return 1;
    }

    cnpypp::npy_save(fortran_path.string(), floats.begin(), {2, 3}, "w", cnpypp::MemoryOrder::Fortran);
    if (!expect(throws_exception<std::invalid_argument>([&fortran_path] {
            (void)leakflow::extras::load_npy(fortran_path);
        }),
            "Fortran-order arrays should be rejected in Phase 16")) {
        return 1;
    }

    if (!expect(throws_exception<std::runtime_error>([] {
            (void)leakflow::extras::load_npy(temp_path("leakflow_missing_numpy_payload.npy"));
        }),
            "missing NumPy path should throw")) {
        return 1;
    }

    std::filesystem::remove(float_path);
    std::filesystem::remove(uint8_path);
    std::filesystem::remove(fortran_path);

    return 0;
}
