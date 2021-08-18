#!/bin/sh

[ -d "noarr-structures" ] || git clone https://github.com/ParaCoToUl/noarr-structures.git

nvcc -O3 -arch=sm_35 -G --expt-relaxed-constexpr -I../../include/ \
         -Inoarr-structures/include/ \
         -I/usr/share/R/include/ \
         -L/usr/lib/R/lib -lR \
         --shared -Xcompiler -fPIC -o matrix_multiply.so matrix_multiply.cu

c++ -O2 -o create_matrix create_matrix.cpp
./create_matrix 32
./create_matrix 64
./create_matrix 96
