#### Noarr tests  <!-- Exclude this line from linear documentation -->
![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20ubuntu-latest%20-%20clang/badge.svg)
![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20ubuntu-latest%20-%20gcc/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20macosl/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-pipelines/workflows/Noarr%20test%20Win/badge.svg)

# Noarr Pipelines

Library that helps programmer with repetitive tasks of setting up computational pipelines for GPGPU CUDA programming.


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


## Documentation

The documentation can be found in the [docs folder](docs).


## Examples

The [examples folder](examples) contains two examples that demonstrate the usage of this library:

- `uppercase`: A small, demonstrative example, showing the producer-consumer pattern. You should be able to understand this example after reading the [Core principles](docs/core-principles.md) section of the documentation.
- `kmeans`: A large example, implementing the k-means algorithm, using almost the entire library, with the cuda extension and with noarr structures.


## Bindings

The [bindings folder](bindings) contains an example project that is experted as a binding to both the R language and the Python language. You can duplicate this folder and modify its contents to package your own GPGPU algorithm.


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
