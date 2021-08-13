#ifndef EXAMPLES_KMEANS_UTILITIES_CPP
#define EXAMPLES_KMEANS_UTILITIES_CPP

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

#include "point_t.hpp"

namespace utilities {

/**
 * Generates random points around given centroids
 * 
 * @param centroids: the centroids around which to generate points
 * @param count: total number of points to generate
 * @param centroid_dispersion: how far away from centroids will generated points be dispersed
 * @param generated_points: here, the generated points will be written
 */
void generate_random_points(
    const std::vector<point_t>& centroids,
    std::size_t count,
    float centroid_dispersion,
    std::vector<point_t>& generated_points
) {
    std::cout << "Generating " << count << " points into "
        << centroids.size() << " clusters..." << std::endl;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    std::default_random_engine generator;
    std::uniform_real_distribution<float> random_dispersion(-centroid_dispersion, centroid_dispersion);
    std::uniform_int_distribution<std::size_t> random_centroid(0, centroids.size() - 1);

    generated_points.resize(count);

    for (std::size_t i = 0; i < count; ++i) {
        point_t p = centroids[random_centroid(generator)];
        p.x += random_dispersion(generator);
        p.y += random_dispersion(generator);
        generated_points[i] = p;
    }

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

    std::cout << "Points generated in " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() <<
        " milliseconds." << std::endl;
}

/**
 * Prints given points as a vertical list to the standard output
 */
void print_points(const std::vector<point_t>& points) {
    for (const point_t& c : points) {
        std::cout << std::setfill(' ') << std::setw(10)
            << std::fixed << std::setprecision(2) << c.x;
        std::cout << ", ";
        std::cout << std::setfill(' ') << std::setw(10)
            << std::fixed << std::setprecision(2) << c.y;
        std::cout << std::endl;
    }
}

/**
 * Computes the squared distance between two points
 */
float squared_distance(point_t a, point_t b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx*dx + dy*dy;
}

/**
 * Computes the mean squared distance between two sets of centroids
 * (the verification metric)
 */
float get_centroids_mean_squared_distance(
    const std::vector<point_t>& computed_centroids,
    const std::vector<point_t>& expected_centroids
) {
    // makes sure each expected centroid has a close corresponding
    // centroid in the computed centroids

    assert(expected_centroids.size() == computed_centroids.size());
    assert(expected_centroids.size() >= 1);

    float total_distance = 0.0f;
    for (point_t expected : expected_centroids) {
        float closest = std::numeric_limits<float>::infinity();
        for (point_t computed : computed_centroids) {
            float dist = squared_distance(expected, computed);
            if (dist < closest)
                closest = dist;
        }
        total_distance += closest;
    }

    return total_distance / expected_centroids.size();
}

/**
 * Counts up how many points are assigned to what centroid and prints the result
 */
void print_assignments_histogram(std::size_t k, const std::vector<std::size_t>& assignments) {
    std::vector<std::size_t> counts(k, 0);

    for (std::size_t assignment : assignments) {
        assert(assignment >= 0
            && "Assignment is negative and thus makes no sense");
        assert(assignment < assignments.size()
            && "Assignment is too large and thus makes no sense");

        counts[assignment] += 1;
    }

    std::cout << "Number of points assigned to computed centroids:" << std::endl;
    for (std::size_t i = 0; i < k; ++i) {
        std::cout << " " << counts[i];
    }
    std::cout << std::endl;
}

/**
 * Checks that the computed centroids are close enough to expected centroids
 * and prints additional information to the standard output
 */
void validate_kmeans_output(
    const std::vector<point_t>& computed_centroids,
    const std::vector<point_t>& expected_centroids,
    const std::vector<std::size_t>& computed_assignments,
    float success_threshold
) {
    float distance = get_centroids_mean_squared_distance(computed_centroids, expected_centroids);

    std::cout << "Average centroid distance: " << distance <<
        " compared to threshold: " << success_threshold << std::endl;

    if (distance < success_threshold) {
        std::cout << "-----------" << std::endl;
        std::cout << "- SUCCESS -" << std::endl;
        std::cout << "-----------" << std::endl;
    } else {
        std::cout << "@@@@@@@@@@@" << std::endl;
        std::cout << "@  ERROR  @" << std::endl;
        std::cout << "@@@@@@@@@@@" << std::endl;
        std::cout << "=> Centroids were found incorrectly." << std::endl;
        std::cout << "But this might be due to unlucky choice of initial centroids." <<
            " Try running the example again to be sure." << std::endl;
    }

    std::cout << std::endl;

    std::cout << "Expected centroids (X Y coordinates):" << std::endl;
    utilities::print_points(expected_centroids);
    std::cout << "Computed centroids (X Y coordinates):" << std::endl;
    utilities::print_points(computed_centroids);
    std::cout << "Order of centroids does not matter." << std::endl;

    std::cout << std::endl;

    print_assignments_histogram(computed_centroids.size(), computed_assignments);
}

} // namespace utilities

#endif
