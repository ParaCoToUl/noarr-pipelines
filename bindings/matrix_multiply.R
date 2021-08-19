#!/usr/bin/env Rscript

matrix_multiply_demo <- function(file, size, repetitions) {
    if(!is.loaded("matrix_multiply_demo")) {
        message("Loading the shared library...")
        if (Sys.info()["sysname"] == "Linux") {
            dyn.load("matrix_multiply.so")
        } else if (Sys.info()["sysname"] == "Windows") {
            dyn.load("matrix_multiply.dll")
        }
    }

    n_matrices <- repetitions

    matrices <- rep(file, n_matrices)
    heights <- rep(size, n_matrices)
    widths <- rep(size, n_matrices)

    res <- .C("matrix_multiply_demo",
        n_matrices=as.integer(n_matrices),
        matrices=as.character(matrices),
        heights=as.integer(heights),
        widths=as.integer(widths))
}

matrix_multiply_demo("data/matrix32.data", 32, 32)
matrix_multiply_demo("data/matrix64.data", 64, 64)
matrix_multiply_demo("data/matrix96.data", 96, 96)
