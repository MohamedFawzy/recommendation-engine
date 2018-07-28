# disable numpy # WARNING:
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy
import pandas as pd
import os

# set path for data
current_working_dir = os.getcwd()
print(current_working_dir)
path = current_working_dir + "/ml-100k/u.data"
print("Reading file ================================>>>\n")
print(path)
column_names = ['user id' , 'item id' , 'rating' , 'timestamp']
df   = pd.read_csv(path, sep ="\t", header= None, names = column_names)

print(type(df))
print("===================\n")
# get first six results of the data frame to have a look at how data seems to be using
print(df.head())
# print columns for dataframe
print(df.columns)
print(df.shape)
