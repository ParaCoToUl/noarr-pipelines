#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>

#include <noarr/pipelines.hpp>
#include <noarr/structures.hpp>

#include "point_t.hpp"
#include "utilities.cpp"

using namespace noarr::pipelines;

/**
 * Computes the distance between two points (eukleidian squared)
 */
static float distance(const point_t& point, const point_t& centroid) {
    float dx = point.x - centroid.x;
    float dy = point.y - centroid.y;
    return (float)(dx*dx + dy*dy);
}

static std::size_t get_nearest_cluster(
    const point_t& point,
    const point_t* centroids,
    std::size_t k
) {
    float minDist = distance(point, centroids[0]);
    std::size_t nearest = 0;
    for (std::size_t i = 1; i < k; ++i) {
        float dist = distance(point, centroids[i]);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }

    return nearest;
}

/**
 * Implements the naive kmeans algorithm using noarr pipelines and noarr structures
 * 
 * @param given_points: vector of points that we want to cluster
 * @param k: number of centroids to cluster into
 * @param refinements: number of refinements (iterations) of the algorithm to perform
 * @param computed_centroids: here, the computed centroids will be written
 * @param computed_assignments: here, the computed assignemnts will be written (cluster indices for each input point)
 */
void kmeans(
    const std::vector<point_t>& given_points,
    std::size_t k,
    std::size_t refinements,
    std::vector<point_t>& computed_centroids,
    std::vector<std::size_t>& computed_assignments
) {
    using PointList = noarr::vector<'i', noarr::array<'d', 2, noarr::scalar<float>>>; // AoS
    // using PointList = noarr::array<'d', 2, noarr::vector<'i', noarr::scalar<float>>>; // SoA

    // hubs only contain structures with a specific size set,
    // so we will name the type of the sized point list structure
    using SizedPointList = decltype(PointList() | noarr::set_length<'i'>(0));

    // TODO: try to use the bag wrapper when working with envelopes
    // TODO: add structures to other hubs as well

    //////////////////////////
    // Define pipeline hubs //
    //////////////////////////

    auto points_hub = Hub<SizedPointList>(
        PointList() | noarr::set_length<'i'>(given_points.size()) | noarr::get_size()
    );
    points_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto assignments_hub = Hub<std::size_t, std::size_t>(sizeof(std::size_t) * given_points.size());
    assignments_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto centroids_hub = Hub<std::size_t, point_t>(sizeof(point_t) * k);
    centroids_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto sums_hub = Hub<std::size_t, point_t>(sizeof(point_t) * k);
    sums_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto counts_hub = Hub<std::size_t, std::size_t>(sizeof(std::size_t) * k);
    counts_hub.allocate_envelope(Device::HOST_INDEX);

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
    auto iterator = LambdaAsyncComputeNode("iterator");

    // a nested code block to define the iterator compute node
    // (the code block is only here for the readablity and is not needed)
    {
        auto& points_link = iterator.link(points_hub.to_peek(Device::HOST_INDEX));
        auto& assignments_link = iterator.link(assignments_hub.to_modify(Device::HOST_INDEX));
        auto& centroids_link = iterator.link(centroids_hub.to_modify(Device::HOST_INDEX));
        auto& sums_link = iterator.link(sums_hub.to_modify(Device::HOST_INDEX));
        auto& counts_link = iterator.link(counts_hub.to_modify(Device::HOST_INDEX));
        
        iterator.initialize([&](){
            // tranfser points data to the hub
            auto& points_envelope = points_hub.push_new_chunk();
            points_envelope.structure = points_envelope.structure | noarr::set_length<'i'>(given_points.size());
            for (std::size_t i = 0; i < given_points.size(); ++i) {
                point_t p = given_points[i];
                points_envelope.structure | noarr::get_at<'i', 'd'>(points_envelope.buffer, i, 0) = p.x;
                points_envelope.structure | noarr::get_at<'i', 'd'>(points_envelope.buffer, i, 1) = p.y;
            }

            // prepare initial centroids
            auto centroids = centroids_hub.push_new_chunk().buffer;
            for (std::size_t i = 0; i < k; ++i)
                centroids[i] = given_points[i];

            // put empty chunks into the remaining hubs
            assignments_hub.push_new_chunk();
            sums_hub.push_new_chunk();
            counts_hub.push_new_chunk();
        });

        /**
         * The iterator can be advanced as long as we have not finished all the refinements yet
         */
        iterator.can_advance([&](){
            return finished_refinements < refinements;
        });

        iterator.advance_async([&](){
            auto& points_envelope = *points_link.envelope;
            auto assignments = assignments_link.envelope->buffer;
            auto centroids = centroids_link.envelope->buffer;
            auto sums = sums_link.envelope->buffer;
            auto counts = counts_link.envelope->buffer;

            for (std::size_t i = 0; i < k; ++i) {
                sums[i].x = sums[i].y = 0;
                counts[i] = 0;
            }

            // TODO: this loop can be turned into a kernel and do all this on cuda
            point_t point;
            for (std::size_t i = 0; i < given_points.size(); ++i) {
                point.x = points_envelope.structure | noarr::get_at<'i', 'd'>(points_envelope.buffer, i, 0);
                point.y = points_envelope.structure | noarr::get_at<'i', 'd'>(points_envelope.buffer, i, 1);

                std::size_t nearest = get_nearest_cluster(point, centroids, k);
                assignments[i] = nearest;
                sums[nearest].x += point.x;
                sums[nearest].y += point.y;
                ++counts[nearest];
            }

            for (std::size_t i = 0; i < k; ++i) {
                if (counts[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.
                centroids[i].x = sums[i].x / counts[i];
                centroids[i].y = sums[i].y / counts[i];
            }
        });

        iterator.post_advance([&](){
            // one additional refinement has just finished
            finished_refinements += 1;
        });

        iterator.terminate([&](){
            // pull the final centroids out
            auto centroids = centroids_hub.peek_top_chunk().buffer;
            computed_centroids.resize(k);
            memcpy(&computed_centroids[0], centroids, sizeof(point_t) * k);

            // assignments are pulled the same way
            auto assignments = assignments_hub.peek_top_chunk().buffer;
            computed_assignments.resize(given_points.size());
            memcpy(&computed_assignments[0], assignments, sizeof(std::size_t) * given_points.size());
        });
    }

    ////////////////////////////////////////////////
    // Set up the scheduler and run to completion //
    ////////////////////////////////////////////////

    // setup pipeline scheduler (give it all pipeline nodes)
    DebuggingScheduler scheduler;
    scheduler.add(points_hub);
    scheduler.add(assignments_hub);
    scheduler.add(centroids_hub);
    scheduler.add(sums_hub);
    scheduler.add(counts_hub);
    scheduler.add(iterator);

    std::cout << "Running kmeans in AoS mode" << std::endl;
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
