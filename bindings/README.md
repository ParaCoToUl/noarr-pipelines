# Bindings to R and Python

This folder gives an example of how to create bindings to R and Python with a code containing noarr-structures and noarr-pipelines.

This allows us to accelerate algorithms written in R and Python with C++ and CUDA (which then helps us to efficiently utilize both CPU and GPU processing power).

The binding is performed by the following steps:

- Creation of a shared dynamic library from C++ or CUDA code
- Dynamic load of the library into R or Python
- Call to an exported function (in C++ marked by `extern "C"`, or `extern "C" __declspec(dllexport)` on Windows)

The steps will be explained in detail in the following two sections (all examples are taken from [matrix_multiply.cu](matrix_multiply.cu) for C++/CUDA, from [matrix_multiply.R](matrix_multiply.R) for R , and from [matrix_multiply.py](matrix_multiply.py) for Python):

## Creating the dynamic library

1. Create a wrapper function(s) for your C++/CUDA code and mark this wrapper function `extern "C"` (or `extern "C" __declspec(dllexport)` on Windows). This can be either used as a specifier in the function definition, or as a specifier for a top-level block. It causes the function to use C linkage (which disables C++ name mangling that adds types of arguments (among other things) to the function's linking name), this allows us to call the function by its name in the dynamic library we make.

    In this example we define a macro to make the `extern "C"` specifier platform-independent:

    ```cpp
    #ifdef _WIN32
    #  define EXPORT extern "C" __declspec(dllexport)
    #else
    #  define EXPORT extern "C"
    #endif
    ```

    When you create the wrapper function, make sure its arguments are supported by the language you are interested in accelerating. The R language is more limiting in this regard than Python (which supports almost arbitrary types)

    - for more info on R's supported types, type `?.C` into an R console
    - for more info on Python's supported types, read the [ctypes](https://docs.Python.org/3/library/ctypes.html) documentation.

2. compile the code into a shared dynamic library (this will be different depending on the compiler you use and the platform). We will describe how it is done for nvcc as our main focus is GPU acceleration using CUDA:

    ```sh
    nvcc -O3 -arch=sm_35 -G --expt-relaxed-constexpr -I "noarr-structures/include/" -I "../include/" --shared -o <LIBRARY_NAME> <SOURCE(S)>
    ```

    - `O3` sets the optimization to the highest level (prioritizing performance)
    - **IMPORTANT:** `arch` sets the target GPU architecture for which we build
    - `--expt-relaxed-constexpr` allows us to use `constexpr` functions in both the HOST code and in the DEVICE code (which is not otherwise possible). It is possible for most `constexpr` functions because they can be precomputed in compile time.
    - `-I "noarr-structures/include/"` and `-I "../include/"`, each add an extra include directory containing source headers
    - **IMPORTANT:** `--shared` specifies we want to create a dynamic library
    - `-o <LIBRARY_NAME>` specifies the name of the outputted file (in this case library)
        - in Linux, the file extension should be `.so`
        - in Windows, the file extension should be `.dll`
    - `<SOURCE(S)>` is a list of source files the library consists of

    For extra information, read the [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options) documentation

## Loading the dynamic library and calling the C code

### R

In R, we load the library by:

```R
dyn.load("<LIBRARY_NAME>")
```

`<LIBRARY_NAME>` is the same as in the build command (there will be different file extensions on different platforms).

We can distinguish the platform the R code runs on by `Sys.info()["sysname"]`, which returns the name of the platform (e.g. `"Linux"` or `"Windows"`)

After the library is loaded, we make calls to the C code by using the `.C` function (the dot is a part of the name). We simply give it the name of the function we want to call and the list of arguments like the following example:

```R
result <- .C("matrix_multiply_demo",
    n_matrices=as.integer(n_matrices),
    matrices=as.character(matrices),
    heights=as.integer(heights),
    widths=as.integer(widths))
```

> This makes a call to the function with the following header (the types have to match, for that see the documentation, otherwise an undefined behavior happens):
>
> ```cpp
> EXPORT
> void matrix_multiply_demo(int *n_matrices, char **matrices,
>                           int *heights, int *widths)
> ```

R integer, character, numeric, etc. vectors are copied into C arrays and then copied back into the corresponding vectors to the returned variable (here stored in `result`). This means that, for example, if the called function (`matric_multiply_demo`) changes a value in `matrices`, the change will be reflected in `result$matrices`. Also mind that a character vector (`matrices`), is passed as a `char **` (array of arrays of characters) and not `char *` (array of characters), R character corresponding to c-string which already is an array of characters (this naming can be slightly counterintuitive).

For more information, type `?.C` to an R console.

### Python

For Python, we will use the standard *[ctypes](https://docs.Python.org/3/library/ctypes.html)* library. So make sure it is imported in your Python code:

```Python
from ctypes import *
```

In Python, we load the library and store it to a variable by:

```Python
variable = CDLL('<LIBRARY_NAME>')
```

`<LIBRARY_NAME>` is the same as in the build command (there will be different file extensions on different platforms).

We can distinguish the platform the Python code runs on by calling `system()` from the standard `platform` library, which returns the name of the platform (e.g. `"Linux"` or `"Windows"`)

After the library has been loaded we make calls to the C functions as if they were Python functions defined in the variable we have loaded the library into, but first, we have to convert the arguments into the `ctypes` library's special types, we will show this on the demonstrational example:

```Python
# this syntax creates 4 arrays (3 integer arrays and 1 c-string array)

# integer array of length `1`, initialized to `repetitions`
n_matrices = (c_int * 1)(repetitions)
# c-string array of length `repetitions`
matrices = (c_char_p * repetitions)()
# integer array of length `repetitions`
heights = (c_int * repetitions)()
# integer array of length `repetitions`
widths = (c_int * repetitions)()

# here we initialize the arrays
for i in range(repetitions):
    matrices[i] = bytes(file, encoding='utf8')
    heights[i] = size
    widths[i] = size

# we pass the 4 arrays as arguments to the C code
matrix_multiply.matrix_multiply_demo(n_matrices, matrices, heights, widths)
```

> This makes a call to the function with the following header (the types have to match, for that see the documentation, otherwise an undefined behavior happens):
>
> ```cpp
> EXPORT
> void matrix_multiply_demo(int *n_matrices, char **matrices,
>                           int *heights, int *widths)
> ```

Note that we use arrays just because R does not support non-array types and we use the same C code for both languages, Python can pass even scalar values (and structures/unions).

For more information, see the [ctypes](https://docs.Python.org/3/library/ctypes.html) documentation.

## How to build the demonstrational dynamic library (for both R and Python)

The following two sections describe how to build the dynamic library on Linux and Windows, this assumes you have installed [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (does not apply to dummy GPU versions).

### Linux environment

Simply run [`./build.sh`](./build.sh) in this folder.

Alternatively, for a version with a dummy GPU simulated on CPU (useful if you do not want to use GPU acceleration), run [`./build_dummy.sh`](./build_dummy.sh).

### Windows environment

For windows, there is an extra requirement of having installed [Visual Studio](https://visualstudio.microsoft.com/cs/).

1. Open a developer command prompt (included in Visual Studio) and make sure you have correctly set variables to build for your machine

    You can set the variables by entering the following command to the developer command prompt:

    ```cmd
    "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    ```

    This script sets variables to build for 64bit system using *Community* version of *Visual Studion 2017*. The command runs the file `vcvarsall.bat` stored in your Visual Studio folder, adjust it to fit your path to Visual Studio and the version you use.

2. Run [`.\build.cmd`](./build.cmd) in this folder.

    Alternatively, for a version with a dummy GPU simulated on CPU (useful if you do not want to use GPU acceleration), run [`.\build_dummy.cmd`](./build_dummy.cmd).

## Demonstration

### How to run the R script

- **On Linux**

    In a shell, run:

    ```sh
    ./matrix_multiply.R
    ```

    or:

    ```sh
    Rscript matrix_multiply.R
    ```

- **On Windows**

    Run the file `matrix_multiply.R` in R.

#### **Through R (regardless of environment)**

Open an R console in this folder and enter the following command:

```R
source("matrix_multiply.R")
```

### How to run the Python demonstration

- **On Linux**

    In a shell, run:

    ```sh
    ./matrix_multiply.py
    ```

    or:

    ```sh
    Python matrix_multiply.py
    ```

- **On Windows**

    In powershell/cmd, run:

    ```ps1
    Python.exe matrix_multiply.py
    ```

### Expected output

The expected output for all demonstrations is in the file [expected.txt](expected.txt).
