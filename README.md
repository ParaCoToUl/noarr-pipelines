#### Noarr tests  <!-- Exclude this line from linear documentation -->
![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20ubuntu-latest%20-%20clang/badge.svg)
![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20ubuntu-latest%20-%20gcc/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20macosl/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20Win/badge.svg)

# Noarr Pipelines

Noarr pipelines is a header-only library, designed for building computational pipelines for GPGPU computing. A pipeline, compared to other computational models, has the advantage of being easily parallelizable among its nodes. In addition, Noarr pipelines aim to be lightweight, extensible, and low-level, to be useful for parallel computation research. Our pipelines can take the form of any generic graph, with nodes arbitrarily connected. Each node is periodically executed, to perform a specific part of the overall computation.


## Using the library

Noarr pipelines is a header-only library, so only include path need to be added. The include path should point to the `/include` folder of this repository.

```cmake
# the CMake line that adds the include directory
target_include_directories(<my-app> PUBLIC <cloned-repo-path>/include)
```

The library requires C++ 17 and the threading library.

```cmake
# tell CMake to use the threading library
find_package(Threads REQUIRED)

# and then link it to your app
target_link_libraries(<my-app> PRIVATE Threads::Threads)
```

> **Note:** You can use the `ExternalProject_Add` CMake command to automatically clone our repository. The `examples/kmeans/CMakeLists.txt` file uses it to include the Noarr structures repository.


## Documentation

The documentation can be found in the [docs folder](docs).


## Examples

The [examples folder](examples) contains two examples that demonstrate the usage of this library:

- `uppercase`: A small, demonstrative example, showing the producer-consumer pattern. You should be able to understand this example after reading the [Core principles](docs/core-principles.md) section of the documentation.
- `kmeans`: A large example, implementing the k-means algorithm, using almost the entire library, with the CUDA extension and with noarr structures.


## Bindings

The [bindings folder](bindings) contains an example project that is exported as a binding to both the R language and the Python language. You can duplicate this folder and modify its contents to package your GPGPU algorithm.


## Running tests

Enter the tests folder (`tests/pipelines` or `tests/cuda-pipelines`). In the terminal (linux bash, windows cygwin or gitbash) run the following commands:

```sh
# create and enter the folder that will contain the build files
mkdir build
cd build

# generates build files for your platform
cmake ..

# builds the project using previously generated build files
cmake --build .

# run the built executable
# (this step differs by platform, this example is for linux)
./test-runner
```


## Future work

Here are listed larger features that would be ideally added into the library in the future:

- **Advanced scheduler**: A scheduler that would attain maximal levels of parallelism within the pipeline.
- **Async compute node thread recycling**: The `AsyncComputeNode` currently creates a new thread for each execution. A less wasteful implementation would reuse only one thread, or possibly utilize a thread pool shared across all `AsyncComputeNode`s.
- **Memory transferrers thread recycling**: Just like the `AsyncComputeNode`, memory transferers currently also create a new thread for each transfers. Recycling a single thread would be less wasteful and may improve performance.
