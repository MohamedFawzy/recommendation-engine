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
path = current_working_dir + "/ml-100k/u.user"
print("Reading file ================================>>>\n")
print(path)
column_names = ['userId' , 'age' , 'gender' , 'occupation' , 'zip code']
df   = pd.read_csv(path, sep ="\t", header= None, names = column_names)



print(type(df))
print("===================\n")
# get first six results of the data frame to have a look at how data seems to be using
print(df.head())
# print columns for dataframe
print("Columns : ", df.columns)
# print shape for dataframe
print("Dataframe shape: ", df.shape)

# total unique number of n_users
n_users = df.userId.unique().shape[0]
print("Unique users")
print(n_users)

# create matrix of zeros with n_users * n_movies to store the ratings in the cell of matrix ratings
user_matrix = np.zeros((n_users))
