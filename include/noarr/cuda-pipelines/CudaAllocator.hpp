#ifndef NOARR_CUDA_PIPELINES_CUDA_ALLOCATOR_HPP
#define NOARR_CUDA_PIPELINES_CUDA_ALLOCATOR_HPP

#include <cuda_runtime.h>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/HostAllocator.hpp"

#include "noarr/cuda-pipelines/NOARR_CUCH.hpp"

namespace noarr {
namespace pipelines {

/**
 * Can allocate and free memory on a CUDA capable device
 */
class CudaAllocator : public MemoryAllocator {
private:
    int _cuda_device;
    Device::index_t _device_index;

public:
    CudaAllocator(int cuda_device, Device::index_t device_index) :
        _cuda_device(cuda_device),
        _device_index(device_index)
    { }

    Device::index_t device_index() const override {
        return _device_index;
    };

    void* allocate(std::size_t bytes) const override {
        void* buffer = nullptr;
        NOARR_CUCH(cudaSetDevice(_cuda_device));
        NOARR_CUCH(cudaMalloc(&buffer, bytes));
        return buffer;
    };

    void deallocate(void* buffer) const override {
        NOARR_CUCH(cudaSetDevice(_cuda_device));
        NOARR_CUCH(cudaFree(buffer));
    };
};

} // pipelines namespace
} // namespace noarr

#endif
