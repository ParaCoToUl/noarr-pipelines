# K-means example

This folder contains a standalone CMake project that demonstrates how one could use noarr structures and noarr pipelines together with CUDA to implement the k-means algorithm.


## Highlights

- The example demonstrates the usage of `Hub`s with and without noarr structures.
- `Hub`s from noarr pipelines handle inter-device data transfers transparently and efficiently.
- The `bag` wrapper from noarr structures lets us easily interact with data in `Hub`s.
- Representing lists of 2D points using noarr structures lets us easily switch between AoS and SoA memory layout and compare the two.
- As noarr structures may have long and cumbersome names, they have been extracted into `include/structures.hpp` and named explicitly to ease their usage.
- Compiling the project in two parts (using `g++` and `nvcc`) yields problems during linking if you use c++ templates. The file `main.cu` (together with `include/structures.hpp`) offers a way to perform explicit template instantiation to remedy the problem.


## Compilation

Compile the example using CMake inside a folder called `build`:

```bash
# create compilation folder and enter it
mkdir build
cd build

# create build files
# (build for release to enable optimizations
# - important for speed of noarr structures)
cmake -D CMAKE_BUILD_TYPE=Release ..

# build the project using build files
cmake --build .

# run the compiled example
./kmeans
```


## Algorithm and its implementation

The algorithm receives:

- `std::vector<point_t> points`: a list of 2D points to cluster
- `std::size_t k`: the number of clusters to create
- `std::size_t refinements`: how many refining iterations to compute

The algorithm returns:

- `std::vector<point_t> centroids`: a list of 2D points that are the centers of their clusters (`k` centroids, because there are `k` clusters)
- `std::vector<std::size_t> assignments`: a list of cluster indices (one for each input point), specifying what point belongs to what cluster

The type `point_t` is a type only used in the API and it represents a 2D point with two `float` members `x` and `y`.

The algorithm starts by choosing random centroids. It then refines these centroids iteratively. During each iteration it:

1. *Recomputes assignments*: For a given point, what is the closest centroid? That is the cluster we now belong to.
2. *Recomputes centroids*: For a given cluster, compute the average of all of its points.

This is implemented in `include/kmeans.cpp` using three CUDA kernels and two helper variables. The helper variables are `sums` and `counts` and they serve the centroid recomputation step. The three kernels are:

1. *Clear sums and counts*: Simply zeroes out our helper variables.
2. *Recompute nearest centroids:* This kernel finds the nearest centroid for each point and updates its assignment. It also adds itself to that cluster's `sum` and `count`, thereby implementing part of the centroid recomputation step.
3. *Reposition centroids:* This kernel computes new centroid positions by taking `sums` and dividing them by `counts`, thereby finalizing the centroid recomputation step.

These kernels are wrapped inside noarr pipelines code that controls their invocation and manages the necessary data. The resulting pipeline is rather simple: it has 5 hubs for all the data (`points`, `centroids`, `assignments`, `sums`, `counts`) and one CUDA compute node (`iterator`) that, when advanced, computes one iteration of the algorithm.

This type of task that iteratively computes on one piece of data is not the typical use-case for noarr pipelines. Pipelines are more suitable for pipelined computation over streams of data. Nonetheless, the example uses noarr pipelines to demonstrate its usage and also to abstract away memory management. This would in theory simplify the process of porting the algorithm to another GPGPU framework.

To learn more about the example, try compiling and running it. The output of the program should be self-explanatory. You can then read through the source code which is also full of comments to guide you.


## Files

- `main.cpp`: The program itself. Parses arguments, generates random points, and then runs kmeans twice (in AoS and SoA mode) and validates its output.
- `main.cu`: Entrypoint for CUDA compilation. It only includes kernel implementations and explicitly instantiates their template functions.
- `include/kmeans.cpp`: The algorithm is implemented here as a function named `kmeans`.
- `include/kernels.cu`: Implementations of kernels and their wrapper functions.

The remaining files are not as crucial:

- `include/kernels.hpp`: Defines the interface that links the `nvcc` compiled part with the `g++` compiled part.
- `include/structures.hpp`: Gives explicit names to certain noarr structures so that they can be easily referenced from other parts of the code. It also acts as part of the interface joining the GPU and CPU parts of the program.
- `include/point_t.hpp`: Defines the `point_t` struct used in the public algorithm API. It is not used in the inner implementation though, noarr structures are used instead.
- `include/utilities.cpp`: Utility functions for generating data and validating the results.


## License

This folder is a part of the noarr pipelines repository and thus belongs under its MIT license as well.
