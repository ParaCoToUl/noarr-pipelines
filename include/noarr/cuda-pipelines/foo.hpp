#ifndef NOARR_CUDA_PIPELINES_FOO_HPP
#define NOARR_CUDA_PIPELINES_FOO_HPP

#include <iostream>
#include <cuda_runtime.h>

namespace noarr {
namespace pipelines {

int foo() {
    void* data;
    
    cudaError_t code = cudaMalloc(&data, 1024);
    
    if (code != cudaSuccess) {
        std::cout << "cudaMalloc failed!" << std::endl;
        return -1;
    }

    std::cout << "cudaMalloc succeeded: " << data << std::endl;
    
    cudaFree(data);
    
    return 42;
}

} // pipelines namespace
} // namespace noarr

#endif
