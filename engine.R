# reading users rating data from csv file
ratings = read.csv("~/Workspace/recommendation_engine/movie_rating.csv")

# data processing and formatting 
movie_ratings = as.data.frame(acast(ratings, title~critic, value.var="rating"))
View(movie_ratings)
