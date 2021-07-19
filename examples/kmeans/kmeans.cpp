#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <random>
#include <noarr/pipelines.hpp>

using namespace noarr::pipelines;

struct point_t {
    float x, y;
};

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

void kmeans(
    const std::vector<point_t>& given_points,
    std::size_t k,
    std::size_t refinements,
    std::vector<point_t>& computed_centroids // output
) {
    // hubs
    auto points_hub = Hub<std::size_t, point_t>(sizeof(point_t) * given_points.size());
    points_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto assignments_hub = Hub<std::size_t, std::uint8_t>(sizeof(std::uint8_t) * given_points.size());
    assignments_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto centroids_hub = Hub<std::size_t, point_t>(sizeof(point_t) * k);
    centroids_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto sums_hub = Hub<std::size_t, point_t>(sizeof(point_t) * k);
    sums_hub.allocate_envelope(Device::HOST_INDEX);
    
    auto counts_hub = Hub<std::size_t, std::size_t>(sizeof(std::size_t) * k);
    counts_hub.allocate_envelope(Device::HOST_INDEX);

    // SIDENOTE: envelopes have not only buffers but also structure, but since
    // we know all the sizes and layout in advance, we don't use it

    // global state
    std::size_t finished_refinements = 0;

    auto iterator = LambdaAsyncComputeNode("iterator");

    /* iterator */ {
        auto& points_link = iterator.link(points_hub.to_peek(Device::HOST_INDEX));
        auto& assignments_link = iterator.link(assignments_hub.to_modify(Device::HOST_INDEX));
        auto& centroids_link = iterator.link(centroids_hub.to_modify(Device::HOST_INDEX));
        auto& sums_link = iterator.link(sums_hub.to_modify(Device::HOST_INDEX));
        auto& counts_link = iterator.link(counts_hub.to_modify(Device::HOST_INDEX));
        
        iterator.initialize([&](){
            // tranfser points data to the hub
            auto points = points_hub.push_new_chunk().buffer;
            memcpy(points, &given_points[0], sizeof(point_t) * given_points.size());

            // prepare initial centroids
            auto centroids = centroids_hub.push_new_chunk().buffer;
            for (std::size_t i = 0; i < k; ++i)
                centroids[i] = points[i];

            // put empty chunks into the remaining hubs
            assignments_hub.push_new_chunk();
            sums_hub.push_new_chunk();
            counts_hub.push_new_chunk();
        });

        iterator.can_advance([&](){
            return finished_refinements < refinements;
        });

        iterator.advance_async([&](){
            auto points = points_link.envelope->buffer;
            auto assignments = assignments_link.envelope->buffer;
            auto centroids = centroids_link.envelope->buffer;
            auto sums = sums_link.envelope->buffer;
            auto counts = counts_link.envelope->buffer;

            for (std::size_t i = 0; i < k; ++i) {
                sums[i].x = sums[i].y = 0;
                counts[i] = 0;
            }

            // TODO: this loop can be turned into a kernel and do all this on cuda
            for (std::size_t i = 0; i < given_points.size(); ++i) {
                std::size_t nearest = get_nearest_cluster(points[i], centroids, k);
                assignments[i] = nearest;
                sums[nearest].x += points[i].x;
                sums[nearest].y += points[i].y;
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

            // TODO: assignments could be pulled the same way
        });
    }

    // setup scheduler
    auto scheduler = DebuggingScheduler();
    scheduler.add(points_hub);
    scheduler.add(assignments_hub);
    scheduler.add(centroids_hub);
    scheduler.add(sums_hub);
    scheduler.add(counts_hub);
    scheduler.add(iterator);

    // run
    scheduler.run();
}


//////////////////
// Support code //
//////////////////

float random_float_between(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

void generate_random_points(
    const std::vector<point_t>& centroids,
    std::size_t count,
    float centroid_dispersion,
    std::vector<point_t>& generated_points // output
) {
    generated_points.clear();

    for (std::size_t i = 0; i < count; ++i) {
        point_t p = centroids[rand() % centroids.size()];
        p.x += random_float_between(-centroid_dispersion, centroid_dispersion);
        p.y += random_float_between(-centroid_dispersion, centroid_dispersion);
        generated_points.push_back(p);
    }
}

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

int main() {
    const std::size_t REFINEMENTS = 1000;
    const std::size_t TOTAL_POINTS = 10000;
    const float CENTROID_DISPERSION = 20.0f;

    srand(time(NULL));

    std::vector<point_t> points;
    std::vector<point_t> expected_centroids = {{0, 0}, {1000, 1000}, {-1000, -2000}};
    std::vector<point_t> computed_centroids;

    std::cout << "Generating points..." << std::endl;
    generate_random_points(
        expected_centroids,
        TOTAL_POINTS,
        CENTROID_DISPERSION,
        points
    );

    std::cout << "Running kmeans..." << std::endl;
    kmeans(points, expected_centroids.size(), REFINEMENTS, computed_centroids);

    std::cout << "Done." << std::endl << std::endl;

    std::cout << "Expected centroids:" << std::endl;
    print_points(expected_centroids);
    std::cout << "Computed centroids:" << std::endl;
    print_points(computed_centroids);

    return 0;
}
