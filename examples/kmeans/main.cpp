#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

// represents a 2D point
// (part of the public API of the kmeans algorithm)
#include "point_t.hpp"

// the kmeans algorithm
#include "kmeans.cpp"

// helper functions for generating data and testing results
#include "utilities.cpp"

int main(int argc, char* argv[]) {

    /*
        Step 1)
        Generate three clusters of 10K points in total.
        Remember the expected centroids of these clusters for later comparison.
     */
    
    std::vector<point_t> points;
    std::vector<point_t> expected_centroids = {{0, 0}, {1000, 1000}, {-1000, -2000}};
    utilities::generate_random_points(
        expected_centroids, // which places to cluster points around
        10000, // total points
        20.0f, // how much to disperse around each centroid
        points // what veriable to write the points to
    );

    std::cout << std::endl;

    /*
        Step 2)
        Run kmeans in AoS mode.
     */

    // TODO .......

    std::cout << "Running kmeans..." << std::endl;
    const std::size_t REFINEMENTS = 1000;
    std::vector<point_t> computed_centroids;
    kmeans(points, expected_centroids.size(), REFINEMENTS, computed_centroids);

    std::cout << "Done." << std::endl;

    std::cout << std::endl;

    std::cout << "Expected centroids:" << std::endl;
    utilities::print_points(expected_centroids);
    std::cout << "Computed centroids:" << std::endl;
    utilities::print_points(computed_centroids);

    return 0;
}
