# disable numpy # WARNING:
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

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
print(df.columns)
# print shape for dataframe
print(df.shape)
# plot dataframe for rating
plt.hist(df['ratings'])
plt.show()

# counts ratings
count_ratings = df.groupby(['ratings'])['userId'].count()
print(count_ratings)
# distribution of movie views
plt.hist( df.groupby(['itemId'])['itemId'].count() )
plt.show

# total number of unique users in dataset
n_users = df.userId.unique().shape[0]
# total number of unique movies in dataset
n_movies = df['itemId'].unique().shape[0]

print(str(n_users) + ' users')
print(str(n_movies) + ' movies')

# create matrix of zeros with n_users * n_movies to store the ratings in the cell of matrix ratings
ratings = np.zeros((n_users, n_movies))
print(ratings)
# foreach tuple in dataframe df extract the information from each column of the row and store it in the rating matrix cell value
for  row in df.itertuples():
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
# create training set and test set
ratings_train, ratings_test = train_test_split(ratings, test_size= 0.33, random_state=42)
# dimensions of the train set, test set
print("Training set shape")
print(ratings_train.shape)

print("Testing set shape")
print(ratings_test.shape)
# predict the user's rating for an item is give by the weighted sum of all other user's ratings for that item.
print("""
###########################################################
#User Based CF                                            #
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
