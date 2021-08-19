#!/usr/bin/env python

from ctypes import *
import platform

if platform.system() == "Linux":
    matrix_multiply = CDLL('./matrix_multiply.so')
elif platform.system() == "Windows":
    matrix_multiply = CDLL('.\matrix_multiply.dll')

def matrix_multiply_demo(file, size, repetitions):
    n_matrices = (c_int * 1)(repetitions)
    matrices = (c_char_p * repetitions)()
    heights = (c_int * repetitions)()
    widths = (c_int * repetitions)()

    for i in range(repetitions):
        matrices[i] = bytes(file, encoding='utf8')
        heights[i] = size
        widths[i] = size

    matrix_multiply.matrix_multiply_demo(n_matrices, matrices, heights, widths)

matrix_multiply_demo('data/matrix32.data', 32, 32)
matrix_multiply_demo('data/matrix64.data', 64, 64)
matrix_multiply_demo('data/matrix96.data', 96, 96)
