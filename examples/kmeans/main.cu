/*
    This file includes all the nvcc compiled parts of kmeans
 */


/////////////////////
// Include kernels //
/////////////////////

#include "kernels.cu"


//////////////////////////////////////////////
// Provide explicit template specialization //
//////////////////////////////////////////////

#include "structures.hpp"

// clear_sums_and_counts_kernel
template void run_clear_sums_and_counts_kernel(
    SizedPointListAoSBag sums_bag,
    std::size_t* counts,
    std::size_t k,
    cudaStream_t stream
);
template void run_clear_sums_and_counts_kernel(
    SizedPointListSoABag sums_bag,
    std::size_t* counts,
    std::size_t k,
    cudaStream_t stream
);

// recompute_nearest_centroids
template void run_recompute_nearest_centroids_kernel(
    SizedPointListAoSBag points_bag,
    SizedPointListAoSBag centroids_bag,
    SizedPointListAoSBag sums_bag,
    std::size_t* assignments,
    std::size_t* counts,
    cudaStream_t stream
);
template void run_recompute_nearest_centroids_kernel(
    SizedPointListSoABag points_bag,
    SizedPointListSoABag centroids_bag,
    SizedPointListSoABag sums_bag,
    std::size_t* assignments,
    std::size_t* counts,
    cudaStream_t stream
);

// reposition_centroids_kernel
template void run_reposition_centroids_kernel(
    SizedPointListAoSBag centroids_bag,
    SizedPointListAoSBag sums_bag,
    std::size_t* counts,
    cudaStream_t stream
);
template void run_reposition_centroids_kernel(
    SizedPointListSoABag centroids_bag,
    SizedPointListSoABag sums_bag,
    std::size_t* counts,
    cudaStream_t stream
);
