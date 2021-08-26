#ifndef NOARR_CUDA_PIPELINES_CUDA_COMPUTE_NODE_HPP
#define NOARR_CUDA_PIPELINES_CUDA_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>
#include <functional>
#include <thread>

#include <cuda_runtime.h>

#include "noarr/pipelines/AsyncComputeNode.hpp"
#include "noarr/cuda-pipelines/NOARR_CUCH.hpp"
#include "noarr/cuda-pipelines/CudaPipelines.hpp"

namespace noarr {
namespace pipelines {

/**
 * Compute node that has its own CUDA stream
 */
class CudaComputeNode : public AsyncComputeNode {
public:
    /**
     * Noarr device index for the CUDA device of the stream
     */
    Device::index_t device_index;
    
    /**
     * Index of the CUDA device that the stream belongs to
     */
    int cuda_device;
    
    /**
     * The CUDA stream to be used for kernel invocations.
     * This compute node automatically synchronizes this stream
     * at the end of advance_cuda.
     */
    cudaStream_t stream = cudaStreamDefault;
    
    CudaComputeNode(Device::index_t device_index) :
        AsyncComputeNode(),
        device_index(device_index),
        cuda_device(CudaPipelines::device_index_to_cuda_device(device_index))
    { }
    
    CudaComputeNode(Device::index_t device_index, const std::string& label) :
        AsyncComputeNode(label),
        device_index(device_index),
        cuda_device(CudaPipelines::device_index_to_cuda_device(device_index))
    { }

protected:

    void advance_async() override {
        __internal__advance_cuda();
    }

    virtual void __internal__advance_cuda() {
        NOARR_CUCH(cudaSetDevice(cuda_device));
        NOARR_CUCH(cudaStreamCreate(&stream));
        
        this->advance_cuda();

        NOARR_CUCH(cudaStreamSynchronize(stream));
        NOARR_CUCH(cudaStreamDestroy(stream));
    }

    virtual void advance_cuda() {
        // override me
    }
};

} // pipelines namespace
} // namespace noarr

#endif
