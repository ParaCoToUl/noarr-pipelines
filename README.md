#### Noarr tests
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


## TODO

- `[x]` docs: core principles
- `[~]` docs: compute node
- `[~]` docs: hub
- `[x]` cuda
- `[x]` kmeans example with cuda
- `[ ]` docs: cuda pipelines
- `[ ]` docs: hardware manager
- `[ ]` python bindings template
- `[~]` R bindings template
- `[ ]` some todo notes in the code and the docs, search for "TODO"
