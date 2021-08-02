#ifndef NOARR_CUDA_PIPELINES_TESTS_CUDA_COMPUTE_NODE_HPP
#define NOARR_CUDA_PIPELINES_TESTS_CUDA_COMPUTE_NODE_HPP

#include <memory>
#include <cuda_runtime.h>

void run_my_kernel(int* items, std::size_t count, cudaStream_t stream);

#endif
