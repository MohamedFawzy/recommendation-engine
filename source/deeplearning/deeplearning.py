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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

idx_to_movie = {}
for row in df.itertuples():
    idx_to_movie[row[1]-1] = row[2]

total_movies = 9000

movies = [0]*total_movies
for i in range(len(movies)):
    if i in idx_to_movie.keys() and len(str(idx_to_movie[i])) == 6:
        movies[i] = (idx_to_movie[i])
movies = filter(lambda imdb: imdb != 0, movies)
total_movies  = len(movies)

URL = [0]*total_movies
IMDB = [0]*total_movies
URL_IMDB = {"url":[],"imdb":[]}
i = 0
for movie in movies:
    (URL[i], IMDB[i]) = get_poster(movie, base_url)
    if URL[i] != base_url+"":
        URL_IMDB["url"].append(URL[i])
        URL_IMDB["imdb"].append(IMDB[i])
    i += 1
# URL = filter(lambda url: url != base_url+"", URL)
df = pd.DataFrame(data=URL_IMDB)

total_movies = len(df)

poster_path = current_working_dir+ "/posters/"
for i in range(total_movies):
    urllib.urlretrieve(df.url[i], poster_path + str(i) + ".jpg")

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage

image = [0]*total_movies
x = [0]*total_movies
for i in range(total_movies):
    image[i] = kimage.load_img(poster_path + str(i) + ".jpg", target_size=(224, 224))
    x[i] = kimage.img_to_array(image[i])
    x[i] = np.expand_dims(x[i], axis=0)
    x[i] = preprocess_input(x[i])

model = VGG16(include_top=False, weights='imagenet')

prediction = [0]*total_movies
matrix_res = np.zeros([total_movies,25088])
for i in range(total_movies):
    prediction[i] = model.predict(x[i]).ravel()
    matrix_res[i,:] = prediction[i]

similarity_deep = matrix_res.dot(matrix_res.T)
norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
similarity_deep = similarity_deep / norms / norms.T
