# load library for recommendation
library(recommenderlab)
# load dataset for jester5k
data("Jester5k")
# sample rating data of the first six users on the first 10 jokes.
head(as(Jester5k, "matrix")[, 1:10])