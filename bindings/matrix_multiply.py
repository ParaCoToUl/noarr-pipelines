#!/usr/bin/env python

from ctypes import *
import platform

# we test what platform the code runs on and then
# we load the dynamic library with the corresponding file extension
if platform.system() == "Linux":
    matrix_multiply = CDLL('./matrix_multiply.so')
elif platform.system() == "Windows":
    matrix_multiply = CDLL('.\matrix_multiply.dll')

def matrix_multiply_demo(file, size, repetitions):
    # this syntax creates 4 arrays (3 integer arrays and 1 c-string array)
    n_matrices = (c_int * 1)(repetitions) # integer array of length `1`, initialized to `repetitions`
    matrices = (c_char_p * repetitions)() # c-string array of length `repetitions`
    heights = (c_int * repetitions)() # integer array of length `repetitions`
    widths = (c_int * repetitions)() # integer array of length `repetitions`

    # here we initialize the arrays
    for i in range(repetitions):
        matrices[i] = bytes(file, encoding='utf8')
        heights[i] = size
        widths[i] = size

    # we pass the 4 arrays to the C code, function `matrix_multiply_demo`
    matrix_multiply.matrix_multiply_demo(n_matrices, matrices, heights, widths)

matrix_multiply_demo('data/matrix32.data', 32, 32)
matrix_multiply_demo('data/matrix64.data', 64, 64)
matrix_multiply_demo('data/matrix96.data', 96, 96)
