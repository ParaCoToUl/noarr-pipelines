matrix_multiply <- function(width, height) {
    data1 <- 1:(height*width)
    data2 <- 1:(height*width)
    data_results <- 1:(height**2)

    res <- .C("matrix_multiply",
        height1=as.integer(height),
        width1=as.integer(width),
        data1=as.integer(data1),
        layout1=as.character("rows"),
        height2=as.integer(width),
        width2=as.integer(height),
        data2=as.integer(data2),
        layout2=as.character("rows"),
        data_results=as.integer(data_results),
        layout1=as.character("rows"))

    print(res$data_results)
}