#!/bin/sh

[ -d "noarr-structures" ] || git clone https://github.com/ParaCoToUl/noarr-structures.git

nvcc -O3 -arch=sm_35 -G --expt-relaxed-constexpr \
         -I "../include/" -I "noarr-structures/include/" \
         --shared -Xcompiler -fPIC -o matrix_multiply.so matrix_multiply.cu
