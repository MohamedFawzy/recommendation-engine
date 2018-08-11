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

