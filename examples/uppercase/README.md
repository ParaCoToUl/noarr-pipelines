# Uppercase example

This folder contains a standalone CMake project that demonstrates the absolute basics of noarr pipelines. It features a pipeline, designed to read a file line by line, capitalize each line and print it to the screen. It demonstrates the producer-consumer pattern that is easy to build using noarr pipelines.

The example consists of two files: 

- `main.cpp`: The entire example, written to use only the CPU. It simulates the usage of a GPU by registering a dummy one. You can easily compile and run this even if you do not have a CUDA-capable device.
- `main.cu`: The entire example, modified to use CUDA.

It does not matter which file you read first, as they are almost identical. Easily start with the CUDA version if you can compile it. Also, comparing the two files, you can see how easily could one switch to a different GPGPU framework when using noarr pipelines.


## Compilation

Compile the example using CMake inside a folder called `build`:

```bash
# create compilation folder and enter it
mkdir build
cd build

# create build files
cmake ..

# build the project using build files
cmake --build . --target uppercase

# run the compiled example on the input file
./uppercase ../input.txt


# IF YOU HAVE CUDA AVAILABLE:

# also build the CUDA target
cmake --build . --target uppercase-cuda

# run the compiled example on the input file
./uppercase-cuda ../input.txt
```


## License

This folder is a part of the noarr pipelines repository and thus belongs under its MIT license as well.
