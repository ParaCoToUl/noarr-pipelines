# Overview of individual tests

- `cuda_compute_node`: Checks that `CudaComputeNode` can be used to run kernels. Also, checks that memory transfers to CUDA devices work.
- `plain_kernel_gcc`: Checks that the testing pipeline can compile and run tests that are compiled by both `gcc` and `nvcc`.
- `plain_kernel_nvcc`: Checks that the testing pipeline can compile and run tests that are compiled by `nvcc` only.
