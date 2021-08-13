#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

// represents a 2D point
// (part of the public API of the kmeans algorithm)
#include "point_t.hpp"

// noarr structures used by the kmeans implementation
#include "structures.hpp"

// the kmeans algorithm
#include "kmeans.cpp"

// helper functions for generating data and testing results
#include "utilities.cpp"

int main(int argc, char* argv[]) {

    /*
        Step 0)
        Parse arguments.
     */

    std::size_t total_points = 10 * 1000;
    std::size_t refinements = 1 * 1000;

    if (argc >= 2) {
        total_points = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        refinements = std::atoi(argv[2]);
    }

    /*
        Step 1)
        Generate three clusters of points.
        Remember the expected centroids of these clusters for later comparison.
     */
    
    std::vector<point_t> points;
    std::vector<point_t> expected_centroids = {{0, 0}, {1000, 1000}, {-1000, -2000}};
    utilities::generate_random_points(
        expected_centroids, // which places to cluster points around
        total_points, // total points
        20.0f, // how much to disperse around each centroid
        points // what veriable to write the points to
    );

    std::cout << std::endl;

    /*
        Step 2)
        Run kmeans in AoS mode.
        Validate its output.
     */

    std::vector<point_t> computed_centroids_aos;
    std::vector<std::size_t> computed_assignments_aos;
    
    kmeans<PointListAoS>(
        points, // input points
        expected_centroids.size(), // k (number of clusters)
        refinements, // number of algorithm iterations to compute
        computed_centroids_aos, // what variable to write the centroids to
        computed_assignments_aos // what variable to write the assignemnts to
    );

    std::cout << std::endl;

    utilities::validate_kmeans_output(
        computed_centroids_aos,
        expected_centroids,
        computed_assignments_aos,
        1.0f // tolerance
    );

    std::cout << std::endl;

    /*
        Step 3)
        Run kmeans in SoA mode.
        Validate its output.
     */

    std::vector<point_t> computed_centroids_soa;
    std::vector<std::size_t> computed_assignments_soa;
    
    kmeans<PointListSoA>(
        points, // input points
        expected_centroids.size(), // k (number of clusters)
        refinements, // number of algorithm iterations to compute
        computed_centroids_soa, // what variable to write the centroids to
        computed_assignments_soa // what variable to write the assignemnts to
    );

    std::cout << std::endl;

    utilities::validate_kmeans_output(
        computed_centroids_soa,
        expected_centroids,
        computed_assignments_soa,
        1.0f // tolerance
    );

    std::cout << std::endl;

    /*
        Step 4)
        Print advanced usage.
     */

    std::cout << "You can customize the behavior in future runs by providing arguments:" << std::endl;
    std::cout << "kmeans [total_points] [refinements]" << std::endl;

    #ifndef NDEBUG
    std::cout << std::endl;
    std::cout << "WARNING: The program has been compiled in debug mode without "
        << "optimizations. For meaningful time measurements compile in release "
        << "mode (see the README compilation section)." << std::endl;
    #endif

    return 0;
}
