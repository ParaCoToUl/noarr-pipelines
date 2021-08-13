#include "my_kernel.hpp"

#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <noarr/cuda-pipelines/NOARR_CUCH.hpp>

__global__ void my_kernel(int* items, std::size_t count) {
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < count)
        items[index] *= items[index];
}

void run_my_kernel(int* items, std::size_t count, cudaStream_t stream) {
    constexpr std::size_t BLOCK_SIZE = 128;

    my_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(items, count);
    NOARR_CUCH(cudaGetLastError());
}
