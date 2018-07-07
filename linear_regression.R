library("MASS")
data("Boston")
set.seed(0)
which_traing <- sample(x = c(TRUE, FALSE), size=nrow(Boston), replace = TRUE, prob = c(0.8, 0.2))
