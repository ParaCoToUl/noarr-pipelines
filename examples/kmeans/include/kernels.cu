#include <limits>
#include <cuda_runtime.h>
#include <noarr/cuda-pipelines/NOARR_CUCH.hpp>

#include "kernels.hpp"

/*
    Each kernel has a corresponding runner function that:
    - lets the kernels be linked from the non-CUDA part of the program
        (kernels cannot be linked by the linker, only functions can)
    - performs minor processing of the given arguments,
        specifies the block size at which to run the kernel,
        and checks for errors after the kernel finishes
 */


//////////////////////////////////
// clear_sums_and_counts_kernel //
//////////////////////////////////

template<typename PointListBag>
__global__ void clear_sums_and_counts_kernel(
    PointListBag sums_bag,
    std::size_t* counts,
    std::size_t k
) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    sums_bag.template at<'i', 'd'>(i, 0) = 0;
    sums_bag.template at<'i', 'd'>(i, 1) = 0;
    counts[i] = 0;
}

template<typename PointListBag>
void run_clear_sums_and_counts_kernel(
    PointListBag sums_bag,
    std::size_t* counts,
    std::size_t k,
    cudaStream_t stream
) {
    constexpr std::size_t BLOCK_SIZE = 128;

    clear_sums_and_counts_kernel<<<
        (k + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream
    >>>(sums_bag, counts, k);
    NOARR_CUCH(cudaGetLastError());
}


////////////////////////////////////////
// recompute_nearest_centroids_kernel //
////////////////////////////////////////

template<typename PointListBag>
__global__ void recompute_nearest_centroids_kernel(
    PointListBag points_bag,
    PointListBag centroids_bag,
    PointListBag sums_bag,
    std::size_t* assignments,
    std::size_t* counts
) {
    std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= points_bag.template get_length<'i'>()) return;

    std::size_t k = centroids_bag.template get_length<'i'>();

    float px = points_bag.template at<'i', 'd'>(p, 0);
    float py = points_bag.template at<'i', 'd'>(p, 1);

    // get nearest centroid index
    std::size_t nearest_centroid_index;
    float nearest_centroid_distance = std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < k; ++c) {
        float cx = centroids_bag.template at<'i', 'd'>(c, 0);
        float cy = centroids_bag.template at<'i', 'd'>(c, 1);
        float dx = px - cx;
        float dy = py - cy;
        float distance = dx*dx + dy*dy;
        if (distance < nearest_centroid_distance) {
            nearest_centroid_distance = distance;
            nearest_centroid_index = c;
        }
    }

    assignments[p] = nearest_centroid_index;
    atomicAdd(&sums_bag.template at<'i', 'd'>(nearest_centroid_index, 0), px);
    atomicAdd(&sums_bag.template at<'i', 'd'>(nearest_centroid_index, 1), py);
    atomicAdd((unsigned long long*)&counts[nearest_centroid_index], 1);
}

template<typename PointListBag>
void run_recompute_nearest_centroids_kernel(
    PointListBag points_bag,
    PointListBag centroids_bag,
    PointListBag sums_bag,
    std::size_t* assignments,
    std::size_t* counts,
    cudaStream_t stream
) {
    constexpr std::size_t BLOCK_SIZE = 128;

    std::size_t point_count = points_bag.template get_length<'i'>();

    recompute_nearest_centroids_kernel<<<
        (point_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream
    >>>(points_bag, centroids_bag, sums_bag, assignments, counts);
    NOARR_CUCH(cudaGetLastError());
}


/////////////////////////////////////
// run_reposition_centroids_kernel //
/////////////////////////////////////

template<typename PointListBag>
__global__ void reposition_centroids_kernel(
    PointListBag centroids_bag,
    PointListBag sums_bag,
    std::size_t* counts
) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= centroids_bag.template get_length<'i'>()) return;
    
    // If the cluster is empty, keep its previous centroid.
    if (counts[i] == 0)
        return;
    
    centroids_bag.template at<'i', 'd'>(i, 0) = sums_bag.template at<'i', 'd'>(i, 0) / counts[i];
    centroids_bag.template at<'i', 'd'>(i, 1) = sums_bag.template at<'i', 'd'>(i, 1) / counts[i];
}

template<typename PointListBag>
void run_reposition_centroids_kernel(
    PointListBag centroids_bag,
    PointListBag sums_bag,
    std::size_t* counts,
    cudaStream_t stream
) {
    constexpr std::size_t BLOCK_SIZE = 128;

    std::size_t k = centroids_bag.template get_length<'i'>();

    reposition_centroids_kernel<<<
        (k + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream
    >>>(centroids_bag, sums_bag, counts);
    NOARR_CUCH(cudaGetLastError());
}
