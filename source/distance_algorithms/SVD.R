sampleMat <- function(n) { i <- 1:n; 1 / outer(i - 1, i, "+") }
original.mat <- sampleMat(9)[, 1:6]
(s <- svd(original.mat))
D <- diag(s$d)
#  X = U D V'
s$u %*% D %*% t(s$v)