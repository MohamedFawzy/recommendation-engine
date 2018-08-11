import pandas as pd
import numpy as np
import scipy
import sklearn

path = "anonymous-msweb.test"

raw_data = pd.read_csv(path,header=None,skiprows=7)
print(raw_data.head())

# create user activity logs
user_activity = raw_data.loc[raw_data[0] != "A"]
# remove unwanted columns from dataset
user_activity = user_activity.loc[:, :1]
# Assign names to the columns of user activity dataframe
user_activity.columns = ['category', 'value']
print(user_activity.head(15))
# get total unique website ids in datasets
print("unique websites")
print(len(user_activity.loc[user_activity['category'] =="V"].value.unique()))
# get total unique users
print('unique users')
print(len(user_activity.loc[user_activity['category'] == "C"].value.unique()))
# create user-item-rating matrix
tmp = 0
nextrow = False
# get last index in dataset
lastindex = user_activity.index[len(user_activity)-1]
print("last index in dataset is ====> ", lastindex)

for index,row in user_activity.iterrows():
    if(index <= lastindex ):
        if(user_activity.loc[index,'category'] == "C"):
            tmp = 0
            userid = user_activity.loc[index,'value']
            user_activity.loc[index,'userid'] = userid
            user_activity.loc[index,'webid'] = userid
            tmp = userid
            nextrow = True
        elif(user_activity.loc[index,'category'] != "C" and nextrow ==  True):
            webid = user_activity.loc[index,'value']
            user_activity.loc[index,'webid'] = webid
            user_activity.loc[index,'userid'] = tmp
            if(index != lastindex and user_activity.loc[index+1,'category'] == "C"):
                nextrow = False
                caseid = 0

print(user_activity.head(30))

user_activity = user_activity[user_activity['category'] == "V" ]
user_activity = user_activity[['userid','webid']]
user_activity_sort = user_activity.sort_values('webid', ascending=True)

sLength = len(user_activity_sort['webid'])

user_activity_sort['rating'] = pd.Series(np.ones((sLength,)),index=user_activity.index)

ratmat = user_activity_sort.pivot(index='userid', columns='webid', values='rating').fillna(0)
ratmat = ratmat.to_dense().as_matrix()

# item profile generation

# filter rows contains first columns as "A"
items = raw_data.loc[raw_data[0] == "A"]
# name the columns
items.columns = ['record', 'webid', 'vote', 'desc', 'url']
# get only two columns so slice dataframe with shape2
items = items[['webid','desc']]
print(items.shape)
