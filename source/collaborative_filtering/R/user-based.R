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
# explore models avaliable and their parameters in recommenderlab package
recommender_model <- recommenderRegistry$get_entries(dataType = "realRatingMatrix")
recommender_model
# build user-based collobrative filtering
recc_model <- Recommender(data = rec_data_train, method= "UBCF")
recc_model
recc_model@model$data
# predictions on test set
n_recommender <- 10
recc_predicit <- predict(object = recc_model, newdata = rec_data_test, n = n_recommender)
recc_predicit
# define list of predicited recommendations :
rec_list <- sapply(recc_predicit@items, function(x){
  colnames(Jester5k)[x]
})

rec_list
