# load library for recommendation
library(recommenderlab)
# load dataset for jester5k
data("Jester5k")
# sample rating data of the first six users on the first 10 jokes.
head(as(Jester5k, "matrix")[, 1:10])
# Building a base recommender model for benchmarking by splitting the data into 80% training data and 20% test data.
# Evaluating the recommender model using a k-fold cross-validation approach model
# Parameter tuning for the recommender model

# Preparing the training data and test data
set.seed(1)
which_train <- sample(x = c(TRUE, FALSE), size = nrow(Jester5k), replace =TRUE, prob = c(0.8, 0.2))
head(which_train)

rec_data_train <- Jester5k[which_train, ]
rec_data_test  <- Jester5k[!which_train, ]
dim(rec_data_train)
dim(rec_data_test)

