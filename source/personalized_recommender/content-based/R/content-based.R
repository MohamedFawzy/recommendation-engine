#1. Generate user profiles.
#2. Generate item profile.
#3. Generate the recommendation engine model. 4. Suggest the top N recommendations.
#4. Suggest the top N recommendations.
user_data = "/Users/mohamedfawzy/Workspace/recommendation_engine/source/personalized_recommender/content-based/R/ml-100k/u.data"
raw_data = read.csv(user_data,sep="\t",header=F)
# adding column names to dataframe
colnames(raw_data) = c("userId", "MovieId", "Ratings", "Timestamp")
ratings = raw_data[, 1:3]
# first five ratings in system
head(ratings)
# see column names
names(ratings)
# see the description of the ratings function
str(ratings)

# load movies data
movies_data = "/Users/mohamedfawzy/Workspace/recommendation_engine/source/personalized_recommender/content-based/R/ml-100k/u.item"

movies = read.csv(movies_data, sep = "|", header = F)
colnames(movies) = c("MovieId","MovieTitle","ReleaseDate","VideoReleaseDate","IMDbURL","Unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller","War","Western")

movies = movies[,-c(2:5)]
View(movies)
names(movies)
str(movies)
ratings = merge(x = ratings, y = movies, by = "MovieId", all.x = TRUE)
View(ratings)
# merge user data , movies data to generate user profile features
ratings = merge(x = ratings, y = movies, by = "MovieId", all.x = TRUE)
View(ratings)
names(ratings)
# set binary rating from 1-3 as 0 and 4-5 as 1
nrat = unlist(lapply(ratings$Ratings, function(x)
{
  
  if(x > 3){ return (1) }
  else { return (0) }
    
}))

ratings = cbind(ratings,nrat)
head(ratings)
apply(ratings[,-c(1:3,23)],2,function(x)table(x))
scaled_ratings = ratings[,-c(3,4)]
scaled_ratings=scale(scaled_ratings[,-c(1,2,21)])
scaled_ratings = cbind(scaled_ratings,ratings[,c(1,2,23)])
head(scaled_ratings)

set.seed(7)
which_train <- sample(x = c(TRUE, FALSE), size = nrow(scaled_ratings),replace = TRUE, prob = c(0.8, 0.2))
model_data_train <- scaled_ratings[which_train, ]
model_data_test <- scaled_ratings[!which_train, ]
dim(model_data_train)
dim(model_data_test)
# use random forrest algorithm to multiple layer classification
library(randomForest)
fit = randomForest(as.factor(nrat)~., data = model_data_train[,-c(19,20)])

predictions <- predict(fit, model_data_test[,-c(19,20,21)], type="class")

cm = table(predictions,model_data_test$nrat)
(accuracy <- sum(diag(cm)) / sum(cm))
(precision <- diag(cm) / rowSums(cm))
recall <- (diag(cm) / colSums(cm))

#extract distinct movieids
totalMovieIds = unique(movies$MovieId)
#see the sample movieids using tail() and head() functions:
#a function to generate dataframe which creates non-rated
#movies by active user and set rating to 0;
nonratedmoviedf = function(userid){
  ratedmovies = raw_data[raw_data$UserId==userid,]$MovieId
  non_ratedmovies = totalMovieIds[!totalMovieIds %in%
                                    ratedmovies]
  df = data.frame(cbind(rep(userid),non_ratedmovies,0))
  names(df) = c("UserId","MovieId","Rating")
  return(df)
}

#let's extract non-rated movies for active userid 943
activeusernonratedmoviedf = nonratedmoviedf(943)


activeuserratings = merge(x = activeusernonratedmoviedf, y = movies, by = "MovieId", all.x = TRUE)


#use predict() method to generate predictions for movie ratings
#by the active user profile created in the previous step.
predictions <- predict(fit, activeuserratings[,-c(1:4)], type="class")
#creating a dataframe from the results
recommend = data.frame(movieId = activeuserratings$MovieId,predictions)
#remove all the movies which the model has predicted as 0 and
#then we can use the remaining items as more probable movies
#which might be liked by the active user.
recommend = recommend[which(recommend$predictions == 1),]