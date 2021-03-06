
library(reshape2)
library(data.table)
library(dplyr)
#data loading
ratings = read.csv("~/Workspace/recommendation_engine/movie_rating.csv")
#data processing and formatting
movie_ratings = as.data.frame(acast(ratings, title~critic,
                                    value.var="rating"))
#similarity calculation
sim_users = cor(movie_ratings[,1:6],use="complete.obs")
#sim_users[colnames(sim_users) == 'Toby']
sim_users[,6]
#predicting the unknown values
#seperating the non rated movies of Toby
rating_critic =
  setDT(movie_ratings[colnames(movie_ratings)[6]],keep.rownames = TRUE)[]
names(rating_critic) = c('title','rating')
titles_na_critic = rating_critic$title[is.na(rating_critic$rating)]
ratings_t =ratings[ratings$title %in% titles_na_critic,]
#add similarity values for each user as new variable
x = (setDT(data.frame(sim_users[,6]),keep.rownames = TRUE)[])
names(x) = c('critic','similarity')
ratings_t = merge(x = ratings_t, y = x, by = "critic", all.x = TRUE)
#mutiply rating with similarity values
ratings_t$sim_rating = ratings_t$rating*ratings_t$similarity
#predicting the non rated titles
result = ratings_t %>% group_by(title) %>%
  summarise(sum(sim_rating)/sum(similarity))
#function to make recommendations
generateRecommendations <- function(userId){
  rating_critic = setDT(movie_ratings[colnames(movie_ratings)[userId]],keep.rownames =TRUE)[]
  names(rating_critic) = c('title','rating')
  titles_na_critic = rating_critic$title[is.na(rating_critic$rating)]
  ratings_t =ratings[ratings$title %in% titles_na_critic,]
  #add similarity values for each user as new variable
  x = (setDT(data.frame(sim_users[,userId]),keep.rownames = TRUE)[])
  names(x) = c('critic','similarity')
  ratings_t = merge(x = ratings_t, y = x, by = "critic", all.x = TRUE)
  #mutiply rating with similarity values
  ratings_t$sim_rating = ratings_t$rating*ratings_t$similarity
  #predicting the non rated titles
  result = ratings_t %>% group_by(title) %>% summarise(sum(sim_rating)/sum(similarity))
  result
  # use mean function per user to get what he will likely most
  mean_ratings = mean(rating_critic$rating, na.rm = T)
  mean_ratings
}

