#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

#include <noarr/structures_extended.hpp>

#include <noarr/pipelines.hpp>
#include <noarr/cuda-pipelines.hpp>
#include <noarr/structures-pipelines.hpp>

#include "point_t.hpp"
#include "kernels.hpp"
#include "utilities.cpp"

using namespace noarr::pipelines;

/**
 * Implements the naive kmeans algorithm using noarr pipelines and noarr structures
 * 
 * @param given_points: vector of points that we want to cluster
 * @param k: number of centroids to cluster into
 * @param refinements: number of refinements (iterations) of the algorithm to perform
 * @param computed_centroids: here, the computed centroids will be written
 * @param computed_assignments: here, the computed assignemnts will be written (cluster indices for each input point)
 */
template<typename PointList>
void kmeans(
    const std::vector<point_t>& given_points,
    std::size_t k,
    std::size_t refinements,
    std::vector<point_t>& computed_centroids,
    std::vector<std::size_t>& computed_assignments
) {
    // hubs only contain structures with a specific size set,
    // so we will explicitly name the type of the sized point list structure
    using SizedPointList = decltype(PointList() | noarr::set_length<'i'>(0));

    // make sure we have the cuda pipelines extension registered
    CudaPipelines::register_extension();

    //////////////////////////
    // Define pipeline hubs //
    //////////////////////////

    // NOTE: all hubs are configured to allocate the exact amount of memory that is needed,
    // since we know exactly how big our input is

    /**
     * Holds input points that we cluster.
     * 
     * Uses noarr structures to represent the point list.
     */
    auto points_hub = Hub<SizedPointList>(
        PointList() | noarr::set_length<'i'>(given_points.size()) | noarr::get_size()
    );
    points_hub.allocate_envelope(Device::HOST_INDEX);
    points_hub.allocate_envelope(Device::DEVICE_INDEX);
    
    /**
     * Holds assignments of input points to computed centroids
     * (an assignment is a centroid index)
     * 
     * Does not use noarr structures, since it is a simple list of numbers.
     * Therefore it does not use the "structure" variable of envelopes and so
     * its type is set to void (void pointer, because void cannot be instantiated).
     * Its buffer item type is set std::size_t because we will treat the buffer as an array.
     */
    auto assignments_hub = Hub<void*, std::size_t>(sizeof(std::size_t) * given_points.size());
    assignments_hub.allocate_envelope(Device::HOST_INDEX);
    assignments_hub.allocate_envelope(Device::DEVICE_INDEX);
    
    /**
     * Holds the list of centroid points, as they are refined
     */
    auto centroids_hub = Hub<SizedPointList>(
        PointList() | noarr::set_length<'i'>(k) | noarr::get_size()
    );
    centroids_hub.allocate_envelope(Device::HOST_INDEX);
    centroids_hub.allocate_envelope(Device::DEVICE_INDEX);
    
    /**
     * Helper buffer that stores point position sums for each cluster
     * to be later divided to obtain the centroid of a given cluster
     */
    auto sums_hub = Hub<SizedPointList>(
        PointList() | noarr::set_length<'i'>(k) | noarr::get_size()
    );
    sums_hub.allocate_envelope(Device::DEVICE_INDEX);
    
    /**
     * Helper buffer that stores point count for each cluster
     * to be used during centroid computation
     */
    auto counts_hub = Hub<void*, std::size_t>(sizeof(std::size_t) * k);
    counts_hub.allocate_envelope(Device::DEVICE_INDEX);


    ///////////////////////////////////
    // Define pipeline compute nodes //
    ///////////////////////////////////

    /**
     * Tracks the finished refinements so that we know when to stop the algorithm
     */
    std::size_t finished_refinements = 0;

    /**
     * The compute node that will compute one iteration of the algorithm each time it is advanced
     */
    auto iterator = LambdaCudaComputeNode("iterator");

    // a nested code block to define the iterator compute node
    // (the code block is only here for the readablity and is not needed)
    {
        // we define all the links to all the hubs
        // the iterator compute node may only be advanced if there are chunks present in all of these hubs
        auto& points_link = iterator.link(points_hub.to_peek(Device::DEVICE_INDEX));
        auto& assignments_link = iterator.link(assignments_hub.to_modify(Device::DEVICE_INDEX));
        auto& centroids_link = iterator.link(centroids_hub.to_modify(Device::DEVICE_INDEX));
        auto& sums_link = iterator.link(sums_hub.to_modify(Device::DEVICE_INDEX));
        auto& counts_link = iterator.link(counts_hub.to_modify(Device::DEVICE_INDEX));

        /**
         * Before the pipeline starts running, put initial data into all hubs
         * (the initialization event method of the iterator compute node is convenient place to put this logic)
         */
        iterator.initialize([&](){
            // tranfser points data to the points hub
            auto& points_envelope = points_hub.push_new_chunk();
            points_envelope.structure = PointList() | noarr::set_length<'i'>(given_points.size());
            auto points_bag = bag_from_envelope(points_envelope);
            for (std::size_t i = 0; i < given_points.size(); ++i) {
                points_bag.template at<'i', 'd'>(i, 0) = given_points[i].x;
                points_bag.template at<'i', 'd'>(i, 1) = given_points[i].y;
            }

            // prepare initial centroids
            // (take first k input points)
            auto& centroids_envelope = centroids_hub.push_new_chunk();
            centroids_envelope.structure = PointList() | noarr::set_length<'i'>(k);
            auto centroids_bag = bag_from_envelope(centroids_envelope);
            for (std::size_t i = 0; i < k; ++i) {
                centroids_bag.template at<'i', 'd'>(i, 0) = given_points[i].x;
                centroids_bag.template at<'i', 'd'>(i, 1) = given_points[i].y;
            }

            // put empty chunks into the remaining hubs
            auto& sums_envelope = sums_hub.push_new_chunk(Device::DEVICE_INDEX);
            sums_envelope.structure = PointList() | noarr::set_length<'i'>(k);
            
            // these two hubs do not use the structure variable
            assignments_hub.push_new_chunk(Device::DEVICE_INDEX);
            counts_hub.push_new_chunk(Device::DEVICE_INDEX);
        });

        /**
         * The iterator can be advanced as long as we have not finished all the refinements yet
         */
        iterator.can_advance([&](){
            return finished_refinements < refinements;
        });

        /**
         * Definition of the "advance" method. It implements one refinement iteration of kmeans.
         * (this method runs in a background thread and does not block the scheduler)
         */
        iterator.advance_cuda([&](cudaStream_t stream){
            auto points_bag = bag_from_link(points_link);
            auto sums_bag = bag_from_link(sums_link);
            auto centroids_bag = bag_from_link(centroids_link);
            auto assignments = assignments_link.envelope->buffer;
            auto counts = counts_link.envelope->buffer;

            run_clear_sums_and_counts_kernel(sums_bag, counts, k, stream);

            run_recompute_nearest_centroids_kernel(
                points_bag, centroids_bag, sums_bag, assignments, counts, stream
            );

            run_reposition_centroids_kernel(centroids_bag, sums_bag, counts, stream);
        });

        /**
         * When the advancement finishes, increment the number of refinements done
         * (this method is executed by the scheduler thread again)
         */
        iterator.post_advance([&](){
            // one additional refinement has just finished
            finished_refinements += 1;

            // NOTE: we could do this in the advance method, but since we access a global
            // variable (possibly accessed by other nodes in the future) it is safer
            // to do it here (in the scheduler thread) to prevent race conditions
        });

        /**
         * When the pipeline terminates, we pull the data from hubs and put it into the output parameters
         * (the termiante event method of our iterator compute node is a convenient place to put this logic)
         */
        iterator.terminate([&](){
            // pull the final centroids out
            auto& centroids_envelope = centroids_hub.peek_top_chunk();
            auto centroids_bag = bag_from_envelope(centroids_envelope);
            computed_centroids.resize(k);
            for (std::size_t i = 0; i < k; ++i) {
                computed_centroids[i].x = centroids_bag.template at<'i', 'd'>(i, 0);
                computed_centroids[i].y = centroids_bag.template at<'i', 'd'>(i, 1);
            }

            // pull the computed assignments out
            auto assignments = assignments_hub.peek_top_chunk().buffer;
            computed_assignments.resize(given_points.size());
            memcpy(&computed_assignments[0], assignments, sizeof(std::size_t) * given_points.size());
        });
    }


    ////////////////////////////////////////////////
    // Set up the scheduler and run to completion //
    ////////////////////////////////////////////////

    // setup pipeline scheduler (give it all pipeline nodes)
    SimpleScheduler scheduler;
    scheduler << points_hub
              << assignments_hub
              << centroids_hub
              << sums_hub
              << counts_hub
              << iterator;

    if (std::is_same<PointList, PointListAoS>()) {
        std::cout << "Running kmeans in AoS mode" << std::endl;
    } else if (std::is_same<PointList, PointListSoA>()) {
        std::cout << "Running kmeans in SoA mode" << std::endl;
    } else {
        std::cout << "Running kmeans in unknown mode" << std::endl;
    }
    std::cout << "==========================" << std::endl;
    std::cout << "Refinements: " << refinements << std::endl;
    std::cout << "Running..." << std::endl;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    // run the pipeline to completion
    scheduler.run();

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

    std::cout << "Done." << std::endl;
    std::cout << "Kmeans executed in " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
        " milliseconds." << std::endl;
}
