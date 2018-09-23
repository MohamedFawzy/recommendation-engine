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
sparsity = float(len(rating_matrix.nonzero()[0]))
sparsity /= (rating_matrix.shape[0] * rating_matrix.shape[1])
sparsity *= 100
print("sparsity : ", sparsity)
