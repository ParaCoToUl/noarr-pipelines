#!/usr/bin/env Rscript

matrix_multiply_demo <- function(file, size, repetitions) {
    # we test what platform the code runs on and then
    # we load the dynamic library with the corresponding file extension
    if(!is.loaded("matrix_multiply_demo")) {
        if (Sys.info()["sysname"] == "Linux") {
            dyn.load("matrix_multiply.so")
        } else if (Sys.info()["sysname"] == "Windows") {
            dyn.load("matrix_multiply.dll")
        }
    }

    n_matrices <- repetitions

    # here we create 3 vectors of the length `repetitions`
    matrices <- rep(file, repetitions)
    heights <- rep(size, repetitions)
    widths <- rep(size, repetitions)

    # we pass the 4 arguments to the C code, function `matrix_multiply_demo`
    res <- .C("matrix_multiply_demo",
        n_matrices=as.integer(n_matrices),
        matrices=as.character(matrices),
        heights=as.integer(heights),
        widths=as.integer(widths))
}

matrix_multiply_demo("data/matrix32.data", 32, 32)
matrix_multiply_demo("data/matrix64.data", 64, 64)
matrix_multiply_demo("data/matrix96.data", 96, 96)
