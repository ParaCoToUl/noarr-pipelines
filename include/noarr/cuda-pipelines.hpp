/*
    This file aggregates includes for the entire "cuda-pipelines" module
 */

// common utilities
#include "noarr/cuda-pipelines/NOARR_CUCH.hpp"

// memory management layer
#include "noarr/cuda-pipelines/CudaAllocator.hpp"
#include "noarr/cuda-pipelines/CudaTransferer.hpp"

// pipeline computation
#include "noarr/cuda-pipelines/CudaComputeNode.hpp"
#include "noarr/cuda-pipelines/LambdaCudaComputeNode.hpp"

// extension registration
#include "noarr/cuda-pipelines/CudaPipelines.hpp"
