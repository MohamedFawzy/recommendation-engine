# reading users rating data from csv file
ratings = read.csv("~/Workspace/recommendation_engine/movie_rating.csv")

# data processing and formatting 
movie_ratings = as.data.frame(acast(ratings, title~critic, value.var="rating"))
View(movie_ratings)

# calculating similraity between users
#
# 1. Extract the titles which Toby has not rated.
# 2. For these titles, separate all the ratings given by other critics.
# 3. Multiply the ratings given for these movies by all critics other than Toby with the similarity values of critics with Toby.
# 4. Sum up the total ratings for each movie, and divide this summed up value with the sum of similarity critic values.

sim_users = cor(movie_ratings[,1:6], use="complete.obs")
View(sim_users)




generateRecommendation <- function(userId){
  # predicting the unkown ratings for users
  rating_critic  = setDT(movie_ratings[colnames(movie_ratings) [userId]],keep.rownames = TRUE)[]
  names(rating_critic) = c('title','rating')
  View(rating_critic)
  
  # extract non-related movies from list
  titles_na_critic = rating_critic$title[is.na(rating_critic$rating)]
  
  ratings_t = ratings[ratings$title %in% titles_na_critic,]
  View(ratings_t)
  
  x = (setDT(data.frame(sim_users[,userId]), keep.rownames = TRUE)[])
  names(x) = c('critic', 'similarity')
  ratings_t = merge(x = ratings_t, y =x , by = "critic", all.x = TRUE)
  View(ratings_t)
  
  ratings_t$sim_rating = ratings_t$rating*ratings_t$similarity
  ratings_t
  
  # sum result set
  result = ratings_t %>% group_by(title) %>% summarise( sum(sim_rating) / sum(similarity) )
  result
  
}
