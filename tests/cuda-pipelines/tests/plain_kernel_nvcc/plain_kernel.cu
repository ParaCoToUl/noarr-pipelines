#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <noarr/cuda-pipelines/NOARR_CUCH.hpp>

template<typename Item>
__global__ void plain_kernel(Item* items) {
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    items[index] = (Item) (index + 1);
}

template<typename Item>
Item run_plain_kernel() {
    constexpr std::size_t ITEMS = 1024;
    constexpr std::size_t BLOCK_SIZE = 128;
    constexpr std::size_t BYTES = sizeof(Item) * ITEMS;
    
    Item* cpu_data;
    Item* gpu_data;

    cpu_data = (Item*) malloc(BYTES);
    NOARR_CUCH(cudaMalloc(&gpu_data, BYTES));

    plain_kernel<Item><<<ITEMS / BLOCK_SIZE, BLOCK_SIZE>>>(gpu_data);
    NOARR_CUCH(cudaGetLastError());

    NOARR_CUCH(cudaDeviceSynchronize());

    NOARR_CUCH(cudaMemcpy(cpu_data, gpu_data, BYTES, cudaMemcpyDeviceToHost));
    
    Item sum = 0;
    for (std::size_t i = 0; i < ITEMS; ++i)
        sum += cpu_data[i];

    free(cpu_data);
    NOARR_CUCH(cudaFree(gpu_data));

    return sum;
}

// NOTE: no explicit template specialization needed,
// since we compile everything with nvcc
