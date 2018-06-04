library(reshape2)
library(data.table)
library(dplyr)
# data loading
ratings = read.csv("~/Workspace/recommendation_engine/movie_rating.csv")
# data processing and formatting
movie_ratings = as.data.frame(acast(ratings, title~critic, value.var = "rating"))
# similarity calculation
sim_users = cor(movie_ratings[,1:6], use="complete.obs")
