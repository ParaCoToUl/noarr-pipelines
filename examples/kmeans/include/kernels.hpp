#ifndef EXAMPLES_KMEANS_KERNELS_HPP
#define EXAMPLES_KMEANS_KERNELS_HPP

#include <cuda_runtime.h>

template<typename PointListBag>
void run_clear_sums_and_counts_kernel(
    PointListBag sums_bag,
    std::size_t* counts,
    std::size_t k,
    cudaStream_t stream
);

template<typename PointListBag>
void run_recompute_nearest_centroids_kernel(
    PointListBag points_bag,
    PointListBag centroids_bag,
    PointListBag sums_bag,
    std::size_t* assignments,
    std::size_t* counts,
    cudaStream_t stream
);

template<typename PointListBag>
void run_reposition_centroids_kernel(
    PointListBag centroids_bag,
    PointListBag sums_bag,
    std::size_t* counts,
    cudaStream_t stream
);

#endif
