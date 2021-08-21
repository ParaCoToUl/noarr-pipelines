#ifndef NOARR_CUDA_PIPELINES_LAMBDA_CUDA_COMPUTE_NODE_HPP
#define NOARR_CUDA_PIPELINES_LAMBDA_CUDA_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>
#include <functional>

#include <cuda_runtime.h>

#include "noarr/cuda-pipelines/CudaComputeNode.hpp"

namespace noarr {
namespace pipelines {

/**
 * A compute node built by passing in lambda expressions
 */
class LambdaCudaComputeNode : public CudaComputeNode {
private:
    std::function<void()> __impl__initialize;
    std::function<bool()> __impl__can_advance;
    std::function<void()> __impl__advance;
    std::function<void()> __impl__advance_cuda;
    std::function<void()> __impl__post_advance;
    std::function<void()> __impl__terminate;

public:
    LambdaCudaComputeNode(const std::string& label, Device::index_t device_index = Device::DEVICE_INDEX) :
        CudaComputeNode(device_index, label),
        __impl__initialize([](){}),
        __impl__can_advance([](){ return true; }),
        __impl__advance([](){}),
        __impl__advance_cuda([](){}),
        __impl__post_advance([](){}),
        __impl__terminate([](){})
    { }

    LambdaCudaComputeNode(Device::index_t device_index = Device::DEVICE_INDEX) :
        LambdaCudaComputeNode(typeid(LambdaAsyncComputeNode).name(), device_index)
    { };

public: // setting implementation
    void initialize(std::function<void()> impl) { __impl__initialize = impl; }
    void can_advance(std::function<bool()> impl) { __impl__can_advance = impl; }
    void advance(std::function<void()> impl) { __impl__advance = impl; }
    void advance_cuda(std::function<void()> impl) { __impl__advance_cuda = impl; }
    void advance_cuda(std::function<void(cudaStream_t)> impl) {
        __impl__advance_cuda = [this, impl](){
            impl(this->stream);
        };
    }
    void post_advance(std::function<void()> impl) { __impl__post_advance = impl; }
    void terminate(std::function<void()> impl) { __impl__terminate = impl; }

protected: // using implementation
    void initialize() override { __impl__initialize(); }
    bool can_advance() override { return __impl__can_advance(); }
    void advance() override { return __impl__advance(); }
    void advance_cuda() override { return __impl__advance_cuda(); }
    void post_advance() override { return __impl__post_advance(); }
    void terminate() override { return __impl__terminate(); }
};

} // pipelines namespace
} // namespace noarr

#endif
