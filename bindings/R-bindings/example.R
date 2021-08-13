#!/usr/bin/env Rscript

# this is an example usage of the library

install.packages('noarr.matrix_0.1.tar.gz', type='source')
library('noarr.matrix')

message("demo 1:")

# multiplies two 5x5 matrices stored by rows
# (the data are filled with consecutive numbers 1,2...)
# and stores the result in a matrix stored by rows
# (returns the data of the result matrix)
matrix_multiply_demo(5, 5, "rows", "rows", "rows")

message("demo 2:")

# the same as above, but with a 1x10 matrix and a 10x1 stored by rows
matrix_multiply_demo(1, 10, "rows", "rows", "rows")

message("demo 3:")

# same as demo 1 and 2, but the second matrix is stored by columns
matrix_multiply_demo(5, 5, "rows", "columns", "rows")
matrix_multiply_demo(1, 10, "rows", "columns", "rows")

message("demo 4:")

# same as demo 1 and 2, but the result matrix is stored by columns
matrix_multiply_demo(5, 5, "rows", "rows", "columns")
matrix_multiply_demo(1, 10, "rows", "rows", "columns")
