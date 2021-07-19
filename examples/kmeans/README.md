# k-means example


**Demonstrates:**

- changing hub dataflow during execution
- in-place modification of a buffer inside a hub


**Description:**

Given a set of 2D points and a number k, find k centroids that cluster the points well. The algorithm starts by randomly choosing centroids. Then two phases repeat 1) cluster assignment 2) centroid update. The number of refinements made is also a parameter.
