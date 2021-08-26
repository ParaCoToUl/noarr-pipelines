#ifndef NOARR_CUDA_PIPELINES_CUDA_PIPELINES_HPP
#define NOARR_CUDA_PIPELINES_CUDA_PIPELINES_HPP

#include <cassert>
#include <cuda_runtime.h>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/HardwareManager.hpp"

#include "noarr/cuda-pipelines/NOARR_CUCH.hpp"
#include "noarr/cuda-pipelines/CudaAllocator.hpp"
#include "noarr/cuda-pipelines/CudaTransferer.hpp"

namespace noarr {
namespace pipelines {

/**
 * Manages the cuda-pipelines extension to noarr-pipelines
 */
class CudaPipelines {
private:
    CudaPipelines() = delete; // static class

public:
    /**
     * Converts CUDA device number to noarr device index
     */
    static Device::index_t cuda_device_to_device_index(int cuda_device) {
        return (Device::index_t) cuda_device;
    }

    /**
     * Converts noarr device index to CUDA device number
     */
    static int device_index_to_cuda_device(Device::index_t device_index) {
        return (int) device_index;
    }

    /**
     * Registers the extension to the default hardware manager
     */
    static void register_extension() {
        register_extension(HardwareManager::default_manager());
    }

    /**
     * Registers the extension to a given hardware manager
     */
    static void register_extension(HardwareManager& manager) {
        int cuda_devices = 0;
        NOARR_CUCH(cudaGetDeviceCount(&cuda_devices));
        assert(cuda_devices > 0 && "There are no cuda-capable devices available.");

        // register individual CUDA devices
        for (int cuda_device = 0; cuda_device < cuda_devices; ++cuda_device) {
            register_gpu(
                manager,
                cuda_device,
                cuda_device_to_device_index(cuda_device)
            );
        }
    }

private:
    static void register_gpu(
        HardwareManager& manager,
        int cuda_device,
        Device::index_t device_index
    ) {
        // allocator
        manager.set_allocator_for(
            device_index,
            std::make_unique<CudaAllocator>(cuda_device, device_index)
        );

        // transferer TO
        manager.set_transferer_for(
            Device::HOST_INDEX, device_index,
            std::make_unique<CudaTransferer>(true, cuda_device)
        );

        // transferer FROM
        manager.set_transferer_for(
            device_index, Device::HOST_INDEX,
            std::make_unique<CudaTransferer>(false, cuda_device)
        );
    }
};

} // pipelines namespace
} // namespace noarr

#endif
