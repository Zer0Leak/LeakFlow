#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <iostream>

namespace {

constexpr int kVectorSize = 16;

__global__ void vector_add(const float* lhs, const float* rhs, float* out, int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = lhs[index] + rhs[index];
    }
}

bool check_cuda(cudaError_t status, const char* label)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << label << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

} // namespace

int main()
{
    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return 1;
    }

    std::cout << "CUDA device count: " << device_count << '\n';

    if (device_count < 1) {
        std::cerr << "No CUDA devices available\n";
        return 1;
    }

    cudaDeviceProp device_properties{};
    if (!check_cuda(cudaGetDeviceProperties(&device_properties, 0), "cudaGetDeviceProperties")) {
        return 1;
    }

    std::cout << "CUDA device 0: " << device_properties.name << '\n';

    std::array<float, kVectorSize> lhs{};
    std::array<float, kVectorSize> rhs{};
    std::array<float, kVectorSize> result{};

    for (int index = 0; index < kVectorSize; ++index) {
        lhs[index] = static_cast<float>(index);
        rhs[index] = static_cast<float>(index * 2);
    }

    float* device_lhs = nullptr;
    float* device_rhs = nullptr;
    float* device_result = nullptr;

    const std::size_t byte_count = sizeof(float) * kVectorSize;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_lhs), byte_count), "cudaMalloc lhs")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_rhs), byte_count), "cudaMalloc rhs")) {
        cudaFree(device_lhs);
        return 1;
    }
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result), byte_count), "cudaMalloc result")) {
        cudaFree(device_lhs);
        cudaFree(device_rhs);
        return 1;
    }

    if (!check_cuda(cudaMemcpy(device_lhs, lhs.data(), byte_count, cudaMemcpyHostToDevice), "cudaMemcpy lhs")
        || !check_cuda(cudaMemcpy(device_rhs, rhs.data(), byte_count, cudaMemcpyHostToDevice), "cudaMemcpy rhs")) {
        cudaFree(device_lhs);
        cudaFree(device_rhs);
        cudaFree(device_result);
        return 1;
    }

    vector_add<<<1, kVectorSize>>>(device_lhs, device_rhs, device_result, kVectorSize);
    if (!check_cuda(cudaGetLastError(), "vector_add launch")
        || !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(device_lhs);
        cudaFree(device_rhs);
        cudaFree(device_result);
        return 1;
    }

    if (!check_cuda(cudaMemcpy(result.data(), device_result, byte_count, cudaMemcpyDeviceToHost), "cudaMemcpy result")) {
        cudaFree(device_lhs);
        cudaFree(device_rhs);
        cudaFree(device_result);
        return 1;
    }

    cudaFree(device_lhs);
    cudaFree(device_rhs);
    cudaFree(device_result);

    for (int index = 0; index < kVectorSize; ++index) {
        const float expected = lhs[index] + rhs[index];
        if (result[index] != expected) {
            std::cerr << "Vector add mismatch at index " << index << '\n';
            return 1;
        }
    }

    std::cout << "Vector add OK" << '\n';
    return 0;
}
