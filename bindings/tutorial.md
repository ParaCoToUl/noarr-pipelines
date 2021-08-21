# Bindings tutorial

This tutorial explains how to create bindings for C++/CUDA code to R or python. Thankfully, the steps to create a dynamic library to be loaded in R/python do not differ between the two languages.

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

    When you create the wrapper function, make sure its arguments are supported by the language you are interested in accelerating. The R language is more limiting in this regard than python (which supports almost arbitrary types)

    - for more info on R's supported types, type `?.C` into an R console
    - for more info on python's supported types, read the [ctypes](https://docs.python.org/3/library/ctypes.html) documentation.

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
    - `<SOURCE(S)>` is a list of source files the library consist of

    For extra information, read the [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options) documentation

## Loading the dynamic library

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

R integer, character, numeric, etc. vectors are copied into C arrays and then copied back into the corresponding vectors to the returned variable (here stored in `result`). This means that, for example, if the called function (`matric_multiply_demo`) changes a value in `matrices`, the change will be reflected in `result$matrices`. Also mind that a character vector (`matrices`), is passed as a `char **` (this naming can be slightly counterintuitive).

For more information, type `?.C` to an R console.

### python

For python we will use the standard *[ctypes](https://docs.python.org/3/library/ctypes.html)* library. So make sure it is imported in your python code:

```python
from ctypes import *
```

In python, we load the library and store it to a variable by:

```python
variable = CDLL('<LIBRARY_NAME>')
```

`<LIBRARY_NAME>` is the same as in the build command (there will be different file extensions on different platforms).

We can distinguish the platform the python code runs on by calling `system()` from the standard `platform` library, which returns the name of the platform (e.g. `"Linux"` or `"Windows"`)

After the library has been loaded we make calls to the C functions as if they were python functions defined in the variable we have loaded the library into, but first, we have to convert the arguments into the `ctypes` library's special types, we will show this on the demonstrational example:

```python
# this syntax creates 4 arrays (3 integer arrays and 1 c-string array)
n_matrices = (c_int * 1)(repetitions)
matrices = (c_char_p * repetitions)()
heights = (c_int * repetitions)()
widths = (c_int * repetitions)()

# here we initialize the arrays
for i in range(repetitions):
    matrices[i] = bytes(file, encoding='utf8')
    heights[i] = size
    widths[i] = size

# we pass the 4 arrays as arguments to the C code
matrix_multiply.matrix_multiply_demo(n_matrices, matrices, heights, widths)
```

Note that we use arrays just because R does not support non-array types and we use the same C code for both languages, python can actually pass scalar values (and even structures/unions).

For more information, see the [ctypes](https://docs.python.org/3/library/ctypes.html) documentation.
