# k-means example


**Demonstrates:**

- changing hub dataflow during execution
- in-place modification of a buffer inside a hub


**Description:**

Given a set of 2D points and a number k, find k centroids that cluster the points well. The algorithm starts by randomly choosing centroids. Then two phases repeat 1) cluster assignment 2) centroid update. The number of refinements made is also a parameter.


## Compilation

Compile the example using cmake inside a folder called `cmake`:

```bash
# create compilation folder and enter it
mkdir cmake
cd cmake

# create build files
cmake ..

# build the project using build files
cmake --build .

# run the compiled example
./kmeans
```
