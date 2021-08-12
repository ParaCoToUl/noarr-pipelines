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

} // namespace utilities

#endif
