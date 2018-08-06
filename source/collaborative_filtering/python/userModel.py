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

###### start model and Dataframe to build knn for users ##########
# set path for data
current_working_dir = os.getcwd()
print(current_working_dir)
path = current_working_dir + "/ml-100k/u.data"
print("Reading file ================================>>>\n")
print(path)
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(path, sep='\t', names=columns)
print(df.head())

# total unique number of n_users
n_users = df.user_id.unique().shape[0]
# total number of unique n_movies
n_movies = df['item_id'].unique().shape[0]
print("===============unique users , movies =======================")
print(n_users, n_movies)


# create matrix of zeros with n_users * n_movies to store the ratings in the cell of matrix ratings
ratings = np.zeros((n_users, n_movies))
print(ratings)
# foreach tuple in dataframe df extract the information from each column of the row and store it in the rating matrix cell value
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
print(type(ratings))
print(ratings)

# data Sparsity
sparsity  = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print 'Sparsity: {:4.2f}%'.format(sparsity)
