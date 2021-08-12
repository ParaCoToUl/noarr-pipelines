width <- 4
height <- 5
data1 <- c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
data2 <- c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
data_results <- c(0:24)


res <- .C("multiply_rows_matrix_by_rows_matrix",
    height=as.integer(height),
    width=as.integer(width),
    data1=as.integer(data1),
    data2=as.integer(data2),
    data_results=as.integer(data_results))