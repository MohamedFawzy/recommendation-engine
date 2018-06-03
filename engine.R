# reading users rating data from csv file
ratings = read.csv("~/Workspace/recommendation_engine/movie_rating.csv")

# data processing and formatting 
movie_ratings = as.data.frame(acast(ratings, title~critic, value.var="rating"))
View(movie_ratings)

# calculating similraity between users
sim_users = cor(movie_ratings[,1:6], use="complete.obs")
View(sim_users)

# predicting the unkown ratings for users
rating_critic  = setDT(movie_ratings[colnames(movie_ratings) [6]],keep.rownames = TRUE)[]
names(rating_critic) = c('title','rating')
View(rating_critic)

# extract non-related movies from list
titles_na_critic = rating_critic$title[is.na(rating_critic$rating)]

ratings_t = ratings[ratings$title %in% titles_na_critic,]
View(result)

x = (setDT(data.frame(sim_users[,6]), keep.rownames = TRUE)[])
names(x) = c('critic', 'similarity')
ratings_t = merge(x = ratings_t, y =x , by = "critic", all.x = TRUE)
View(ratings_t)

