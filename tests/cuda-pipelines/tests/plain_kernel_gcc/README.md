Tests that a plain CUDA kernel can be compiled and run. Does not test noarr at all; it tests that tests themselves can be compiled and run.

This folder also acts as a template for other CUDA tests.

This **gcc** version of the test has the `_test.cpp` portion compiled via g++ and the kernel portion compiled via nvcc.
