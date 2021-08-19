@echo off
IF NOT exist noarr-structures (
    git clone https://github.com/ParaCoToUl/noarr-structures.git
)

cl /O2 /LD /EHsc /I noarr-structures\include\ /I ..\include\ matrix_multiply_dummy.cpp /link /out:matrix_multiply.dll
