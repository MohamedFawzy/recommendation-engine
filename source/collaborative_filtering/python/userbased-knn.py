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
