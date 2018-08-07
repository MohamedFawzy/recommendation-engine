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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


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

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],size=10,replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings]  = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test

train, test = train_test_split(ratings)

def cosine_similarity(ratings, kind='user', epslion=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epslion
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epslion
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

user_similarity = cosine_similarity(train, kind='user')
item_similarity = cosine_similarity(train, kind='item')
print("item similarity")
print item_similarity[:4, :4]
print("user similarity")
print user_similarity[:4, :4]

def predict_similarity(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

item_prediction = predict_similarity(train, item_similarity, kind='item')
user_prediction = predict_similarity(train, user_similarity, kind='user')

print 'User-based CF MSE: ' + str(get_mse(user_prediction, test))
print 'Item-based CF MSE: ' + str(get_mse(item_prediction, test))
# KNN for users, items
def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

    return pred

pred = predict_topk(train, user_similarity, kind='user', k=40)
print 'Top-k User-based CF MSE: ' + str(get_mse(pred, test))

pred = predict_topk(train, item_similarity, kind='item', k=40)
print 'Top-k Item-based CF MSE: ' + str(get_mse(pred, test))

# tunning the parameter of k
k_array = [5, 15, 30, 50, 100, 200]
user_train_mse = []
user_test_mse = []
item_test_mse = []
item_train_mse = []


for k in k_array:
    user_pred = predict_topk(train, user_similarity, kind='user', k=k)
    item_pred = predict_topk(train, item_similarity, kind='item', k=k)

    user_train_mse += [get_mse(user_pred, train)]
    user_test_mse  += [get_mse(user_pred, test)]

    item_train_mse += [get_mse(item_pred, train)]
    item_test_mse  += [get_mse(item_pred, test)]


# plot accuracy for tunning k
pal = sns.color_palette("Set2", 2)

plt.figure(figsize=(8, 8))
plt.plot(k_array, user_train_mse, c=pal[0], label='User-based train', alpha=0.5, linewidth=5)
plt.plot(k_array, user_test_mse, c=pal[0], label='User-based test', linewidth=5)
plt.plot(k_array, item_train_mse, c=pal[1], label='Item-based train', alpha=0.5, linewidth=5)
plt.plot(k_array, item_test_mse, c=pal[1], label='Item-based test', linewidth=5)
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=16);
plt.yticks(fontsize=16);
plt.xlabel('k', fontsize=30);
plt.ylabel('MSE', fontsize=30);
plt.show()
