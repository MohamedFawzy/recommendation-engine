# load library for recommendation
library(recommenderlab)
# load dataset for jester5k
data("Jester5k")
# sample rating data of the first six users on the first 10 jokes.
head(as(Jester5k, "matrix")[, 1:10])
# set 80% from dataset for training and 20% to testing
which_train <- sample(x = c(TRUE, FALSE), size = nrow(model_data),replace = TRUE, prob = c(0.8, 0.2))
class(which_train)
head(which_train)
# generate training set

model_data_train <- model_data[which_train, ]
dim(model_data_train)
# generate test data
model_data_test <- model_data[!which_train, ]
dim(model_data_test)
# evaluate the model
n_fold <- 4
items_to_keep <- 15
rating_threshold <- 3
eval_sets <- evaluationScheme(data = model_data, method="cross-validation", k = n_fold, given = items_to_keep, goodRating = rating_threshold)
size_sets <- sapply(eval_sets@runsTrain, length)
size_sets
model_to_evaluate <- "IBCF"
model_parameters <- NULL
getData(eval_sets, "train")
eval_recommender <- Recommender(data = getData(eval_sets, "train"), method= model_to_evaluate, parameter = model_parameters)
items_to_recommend <- 10
eval_prediction <- predict(eval_recommender, newdata = getData(eval_sets, "known"), n = items_to_recommend, type="ratings")
class(eval_prediction)

# train model on traning data
model_to_evaluate <- "IBCF"
# number of nearest items to be checked
model_parameters <- list(k = 30)
# building recommendation engine using training data and item based CF
model_recommender <- Recommender(data = model_data_train , method = model_to_evaluate, parameter = model_parameters)
model_recommender
# extract model details
model_details = getModel(model_recommender)
str(model_details)
# generate recommendations for test data based on training data
# number of items to be recommended
items_to_recommend <- 10
# predict unkown ratings on test set using predict function
model_predicition <- predict(object = model_recommender, newdata= model_data_test, n = items_to_recommend)
model_predicition
print(class(model_predicition))
slotNames(model_predicition)
# get predictions generated for the first user
model_predicition@items[[1]]
# add item labels to each of the predictions
recc_user_1 <- model_predicition@items[[1]]
jokes_user_1 <- model_predicition@itemLabels[recc_user_1]
jokes_user_1

