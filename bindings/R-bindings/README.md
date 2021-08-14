# Bindings to the R language (demo)

This folder gives an example how to create C/C++ bindings to R with a code containing noarr-structures and demonstrates the usage of the created R (CRAN) package.

This demonstration R package exports the function `matrix_multiply_demo(height, width, layout1, layout2, layout_result)` which takes 5 arguments:

1. height: the height of the left matrix (and the width of the right one)
2. width: the width of the left matrix (and the height of the right one)
3. layout1: the layout of the left matrix
4. layout2: the layout of the right matrix
5. layout_result: the layout of the result matrix

All layouts are either `"rows"` or `"columns"` and they indicate how the matrices shall be stored.

`matrix_multiply_demo` then creates two data blobs of the size `width * height` for the left and the right matrix and fills those blobs with the numbers `1:(width * height)`. Then it creates a blob for the result matrix and runs the multiplication (all blobs are represented according to the layouts given).

## How to build

On a linux environment, simply clone the repository containing this folder and run [./pack_cran.sh](./pack_cran.sh).

## How to use

In a R repl console run the following:

```R
# to install the package
install.packages('noarr.matrix_0.1.tar.gz', type='source')

# to use the package
library('noarr.matrix')
```

Make sure `noarr.matrix_0.1.tar.gz` points to the file created by the build script.

The usage is demonstrated in full (from installation to running the `matrix_multiply_demo` function) in the file [example.R](example.R).

On a linux environment, simply run:

```sh
./example.R
```

or

```sh
Rscript example.R
```

This example demonstrates the usefulness of noarr-structures in implementing algorithms using data structures with their layout abstracted away (and thus allowing this implementations run on different data structures). For the implementation, see [src/noarr_matrix.cpp](src/noarr_matrix.cpp) (the main functionality is in the first overload of `matrix_multiply_impl`; there are multiple `matrix_multiply_impl` overloads to facilitate multiple switches from a runtime variable into a distinct type so we can set each matrix to a different layout)
