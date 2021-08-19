# Bindings to R and python (demo)

This folder gives an example how to create bindings to R and python with a code containing noarr-structures and noarr-pipelines.

This allows us to accelerate algorithms written in R and python with C++ and CUDA (which then helps us to efficiently utilize both CPU and GPU processing power).

The binding is performed by the following steps

- Creation of a shared dynamic library from C++ or CUDA code
- Dynamic load of the library into R or python
- Call to an exported function (in C++ marked by `extern "C"`, or `extern "C" __declspec(dllexport)` on Windows)

## How to build the dynamic library (for both R and python)

The following two sections describe how to build the dynamic library on Linux and Windows, this assumes you have installed [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (does not apply to dummy GPU versions).

### Linux environment

Simply run [./build.sh](./build.sh) in this folder.

Alternatively, for a version with a dummy GPU simulated on CPU (useful if you do not want to use GPU acceleration), run [./build_dummy.sh](./build_dummy.sh).

### Windows environment

For windows, there is an extra requirement of having installed [Visual Studio](https://visualstudio.microsoft.com/cs/).

1. Open developer command prompt (included in Visual Studio) and make sure you have correctly set variables to build for your machine

    You might want to do so by entering the following CMD command:

    ```cmd
    C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    ```

    This script sets variables to build for 64bit system using *Community* version of *Visual Studion 2017*. Adjust it to fit your path to Visual Studio and the version you use.

2. Run [.\build.cmd](./build.cmd) in this folder.

    Alternatively, for a version with a dummy GPU simulated on CPU (useful if you do not want to use GPU acceleration), run [.\build_dummy.cmd](./build_dummy.cmd).

## How to run the R demonstration

### Linux

In a shell, run:

```sh
./matrix_multiply.R
```

or:

```sh
Rscript matrix_multiply.R
```

### Windows

Run the file `matrix_multiply.R` in R.

### Through R (regardless of environment)

```R
source("matrix_multiply.R")
```

## How to run the python demonstration

### Linux

In a shell, run:

```sh
./matrix_multiply.py
```

or:

```sh
python matrix_multiply.py
```

### Windows

In powershell/cmd, run:

```ps1
python.exe matrix_multiply.py
```

## Expected output for demonstrations

The expected output for all demonstrations is in the file [expected.txt](expected.txt).
