#ifndef NOARR_CUDA_PIPELINES_CUDA_TRANSFERER_HPP
#define NOARR_CUDA_PIPELINES_CUDA_TRANSFERER_HPP

#include <cuda_runtime.h>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/MemoryTransferer.hpp"

#include "noarr/cuda-pipelines/NOARR_CUCH.hpp"

namespace noarr {
namespace pipelines {

/**
 * Can allocate and free memory on a CUDA capable device
 * 
 * uploading = to the CUDA device
 * downloading = from the CUDA device
 */
class CudaTransferer : public MemoryTransferer {
private:
    bool is_uploading;
    int cuda_device;

public:
    CudaTransferer(bool is_uploading, int cuda_device) :
        is_uploading(is_uploading), cuda_device(cuda_device)
    { }

    virtual void transfer(
        void* from,
        void* to,
        std::size_t bytes,
        std::function<void()> callback
    ) const override {
        // NOTE: custom thread and custom CUDA stream to not block anything else
        std::thread t([this, from, to, bytes, callback](){
            // use the proper device
            NOARR_CUCH(cudaSetDevice(cuda_device));

            // create a stream
            cudaStream_t stream;
            NOARR_CUCH(cudaStreamCreate(&stream));

            // start the transfer
            cudaMemcpyKind kind = is_uploading ?
                cudaMemcpyKind::cudaMemcpyHostToDevice :
                cudaMemcpyKind::cudaMemcpyDeviceToHost;
            NOARR_CUCH(cudaMemcpyAsync(to, from, bytes, kind, stream));

            // wait for it to finish and call back
            NOARR_CUCH(cudaStreamSynchronize(stream));
            callback();

            // clean up
            NOARR_CUCH(cudaStreamDestroy(stream));
        });
        t.detach();
    }
};

} // pipelines namespace
} // namespace noarr

#endif
