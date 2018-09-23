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

# set path for data
current_working_dir = os.getcwd()
print(current_working_dir)
user_data_path = current_working_dir + "/ml-100k/u.data"
print("Reading file ================================>>>\n")
print(user_data_path)
user_data_column_names = ['userId' , 'movieId' , 'ratings' , 'timestamp']
users_data   = pd.read_csv(user_data_path, sep ="\t", header= None, names = user_data_column_names)
print(users_data.head(5))
# read items data
item_data_path = current_working_dir+"/ml-100k/u.item"
item_data_column_names = ["movieId","MovieTitle","ReleaseDate","VideoReleaseDate","IMDbURL","Unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller","War","Western"]
items_data = pd.read_csv(item_data_path , sep="|", header=None, names = item_data_column_names)
print(items_data.head(5))
# merge users and movies then build matrix
df = pd.merge(users_data, items_data, on='movieId')
rating_matrix = np.zeros((df.userId.unique().shape[0], max(df.movieId)))
for row in df.itertuples():
    rating_matrix[row[1]-1, row[2]-1] = row[3]
rating_matrix = rating_matrix[:,:9000]
print("rating matrix\n\n")
print(rating_matrix)
sparsity = float(len(rating_matrix.nonzero()[0]))
sparsity /= (rating_matrix.shape[0] * rating_matrix.shape[1])
sparsity *= 100
print("sparsity : ", sparsity)
# split rating_matrix to training set and test sets
train_matrix = rating_matrix.copy()
test_matrix  = np.zeros(rating_matrix.shape)
for i in xrange(rating_matrix.shape[0]):
    rating_idx = np.random.choice(rating_matrix[i,:].nonzero()[0],size=10,replace=True)
    train_matrix[i,rating_idx] = 0.0
    test_matrix[i,rating_idx] = rating_matrix[i, rating_idx]

# cosine simiarlity among users
simiarlity_user = train_matrix.dot(train_matrix.T) + 1e-9
norms = np.array([np.sqrt(np.diagonal(simiarlity_user))])
simiarlity_user = (simiarlity_user / (norms * norms.T))

simiarlity_movie = train_matrix.T.dot(train_matrix) + 1e-9
norms = np.array([np.sqrt(np.diagonal(simiarlity_movie))])
simiarlity_movie = (simiarlity_movie / (norms * norms.T))
from sklearn.metrics import mean_squared_error
prediction = simiarlity_user.dot(train_matrix) / np.array([np.abs(simiarlity_user).sum(axis=1)]).T
prediction = prediction[test_matrix.nonzero()].flatten()
test_vector = test_matrix[test_matrix.nonzero()].flatten()
mse = mean_squared_error(prediction, test_vector)

print 'MSE = ' + str(mse)

import requests
import json
from IPython.display import Image
from IPython.display import display
from IPython.display import HTML

idx_to_movie = {}
for row in df.itertuples():
    idx_to_movie[row[1]-1] = row[2]
idx_to_movie
k = 6
idx = 0
movies = [ idx_to_movie[x] for x in np.argsort(simiarlity_movie[idx,:])[:-k-1:-1] ]
movies = filter(lambda imdb: len(str(imdb)) == 6, movies)
n_display = 5
URL = [0]*n_display
IMDB = [0]*n_display
i = 0
for movie in movies:
    (URL[i], IMDB[i]) = get_poster(movie, base_url)
    i += 1

images = ''
for i in range(n_display):
    images += "<img style='width: 100px; margin: 0px; \
                float: left; border: 1px solid black;' src='%s' />" \
                % URL[i]

display(HTML(images))
