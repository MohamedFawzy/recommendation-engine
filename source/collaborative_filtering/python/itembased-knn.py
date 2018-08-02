# disable numpy # WARNING:
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import sklearn


# set path for data
current_working_dir = os.getcwd()
print(current_working_dir)
path = current_working_dir + "/ml-100k/u.data"
print("Reading file ================================>>>\n")
print(path)
column_names = ['userId' , 'itemId' , 'ratings' , 'timestamp']
df   = pd.read_csv(path, sep ="\t", header= None, names = column_names)



print(type(df))
print("===================\n")
# get first six results of the data frame to have a look at how data seems to be using
print(df.head())
# print columns for dataframe
print("Columns : ", df.columns)
# print shape for dataframe
print("Dataframe shape: ", df.shape)
# plot dataframe for rating
count_ratings = df.groupby(['ratings'])['userId'].count()
print("Count ratings\n ", count_ratings);

# total unique number of n_users
n_users = df.userId.unique().shape[0]
# total number of unique n_movies
n_movies = df['itemId'].unique().shape[0]
print("===============unique users , movies =======================")
print(n_users, n_movies)


# create matrix of zeros with n_users * n_movies to store the ratings in the cell of matrix ratings
ratings = np.zeros((n_users, n_movies))
print(ratings)
# foreach tuple in dataframe df extract the information from each column of the row and store it in the rating matrix cell value
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
print(type(ratings))
# get shape for the array of count_ratings
print(ratings.shape)
# sample data for how ratings looks like
print(ratings)

# get sparsity in the dataset
# Hint sparsity represent the ratings exist in dataset e.g if we have only 6.3% that means only 6.3% from the dataset has ratings and others has zeros
# Hint zeros means are empty rating
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print('Sparsity: {:4.2}%'.format(sparsity))
# create training set and test set with values 0.33 for test dataset and 42% as Training dataset
ratings_train, ratings_test = train_test_split(ratings, test_size= 0.33, random_state=42)
# dimensions of the train set, test set
print("Training set shape")
print(ratings_train.shape)

print("Testing set shape")
print(ratings_test.shape)

# predict the user's rating for an item is give by the weighted sum of all other user's ratings for that item.
print("""
###########################################################
# Item Based CF with knn                                            #
# 1- Creating similarity matrix between n_users using     #
# cosine similarity.                                      #
#                                                         ###############
# 2- Prediciting unkown rating for item i                 ############################
# for an active user u by calcauting                      ##############################
# the weighted sum of all the users for the item          #####################################
#                                                         ##########################################
# 3- Recommending the new items to the user               ##############################################
#                                                         ################################################
#                                                         ###################################################
###########################################################
""")

# KNN part
k = 10
neighbor = NearestNeighbors(k, 'cosine')
# fit training data to KNN
neighbor.fit(ratings_train.T)
#Calculate the top five similar users for each user and their similarity values, that is the distance values between each pair of users
top_k_distances, top_k_movies = neighbor.kneighbors(ratings_train.T, return_distance=True)
print("Shape==============>", top_k_movies.shape, top_k_distances.shape)
# top five users similar to user 1
print(top_k_movies[0])
# choose only top five users for each user and use their rating information
#while prediciting the ratings using the weighted sum of all of the ratings of these top five similar users
movie_pred_k = np.zeros(ratings_train.shape)
for i in range(ratings_train.T.shape[0]):
       movie_pred_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_movies][i]) /np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

print(movie_pred_k.shape)
print(movie_pred_k)

# error function for the model
def get_mse(pred, actual):
    # ignore nonzeros items
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# model accuracy
print("MSE for Training")
print(get_mse(movie_pred_k, ratings_train))
print("MSE for testing")
print(get_mse(movie_pred_k, ratings_test))