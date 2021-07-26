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


## Running tests and examples

Enter the desired folder (e.g. `examples/kmeans`, `tests/pipelines`, ...). In the terminal (linux bash, windows cygwin or gitbash) run the following commands:

```sh
# generates build files for your platform
cmake .

# builds the project using previously generated build files
cmake --build .

# run the built executable
# (this step differs by platform, this example is for linux)
./kmeans || ./test-runner
```

On MFF gpulab, prefix all commands with `srun` to run them on slurm. And for
cuda code, prefix with e.g. `srun -p volta-hp --gpus=1` to have a GPU available.
More info here: https://gitlab.mff.cuni.cz/ksi/clusters


## TODO

- `[x]` docs: core principles
- `[x]` docs: compute node
- `[ ]` docs: hub
- `[ ]` cuda
- `[ ]` kmeans example with cuda
- `[ ]` docs: cuda pipelines
- `[ ]` docs: hardware manager
- `[ ]` python bindings template
- `[ ]` R bindings template
- `[ ]` some todo notes in the code and the docs, search for "TODO"
