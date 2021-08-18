@echo off
IF exist noarr-structures (
    git clone https://github.com/ParaCoToUl/noarr-structures.git
)

nvcc -O3 -arch=sm_35 -G --expt-relaxed-constexpr -I "noarr-structures/include/" -I "../../include/" --shared -Xcompiler -fPIC -o matrix_multiply.dll matrix_multiply.cu
