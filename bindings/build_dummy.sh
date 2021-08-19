#!/bin/sh

[ -d "noarr-structures" ] || git clone https://github.com/ParaCoToUl/noarr-structures.git

c++ -O3 -I "../include/" -I "noarr-structures/include/" \
        --shared -fPIC -o matrix_multiply.so matrix_multiply_dummy.cpp
