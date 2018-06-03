# reading users rating data from csv file
ratings = read.csv("~/Workspace/recommendation_engine/movie_rating.csv")

# data processing and formatting 
movie_ratings = as.data.frame(acast(ratings, title~critic, value.var="rating"))
View(movie_ratings)

# calculating similraity between users
sim_users = cor(movie_ratings[,1:6], use="complete.obs")
View(sim_users)
