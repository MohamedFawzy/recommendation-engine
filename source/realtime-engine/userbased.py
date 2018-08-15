# load  user ratings dataset
# author: MohamedFawzy
# email: mfawzy22@gmail.com
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
# create spark context app
sc = SparkContext(appName="UserBase")
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
data = sc.textFile("/var/www/html/ml-100k/u.data")
print("====================type===================")
print(type(data))
print("===================type end=========================")
print("===============count========================")
print(data.count())
print("===================count end=========================")
print("===============first row========================")
print(data.first())
print("===============first row end========================")
print("===============first five rows========================")
print(data.take(5))
print("===============end five rows========================")
# get ratings foreach row and remove timestamp column from mapping
ratings = data.map(lambda l: l.split('\t')).map(lambda l:Rating(int(l[0]), int(l[1]), float(l[2])))
print("===============ratings type========================")
print(type(ratings))
print("===============end ratings type========================")
print("===============first five ratings=======================")
print(ratings.take(5))
print("===============first five ratings end=======================")
# get total number of unique users, products
df = spark.createDataFrame(ratings, ["user", 'product'])
df.select("user").distinct().show(5)
df.select("user").distinct().count()
df.select("product").distinct().show(5)
df.select("product").distinct().count()
# number of rated products by each user pick top five
df.groupBy("user").count().show(5)
df.groupBy("rating").count().show()
import numpy as np

# ratings from 1..5 with distribution of number of counts
#import matplotlib.pyplot as plt
# n_groups = 5
# x = df.groupBy("rating").count().select('count')
# xx = x.rdd.flatMap(lambda x: x).collect()
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 1
# opacity = 0.4
# rects1 = plt.bar(index, xx, bar_width, alpha=opacity, color='b', label='ratings')
# plt.xlabel('ratings')
# plt.ylabel('Counts')
# plt.title('Distribution of ratings')
# plt.xticks(index + bar_width, ('1.0', '2.0', '3.0', '4.0', '5.0'))
# plt.legend()
# plt.tight_layout()
# plt.show()


# Statistics of ratings per user
df.groupBy("user").count().select("count").describe().show()
#Individual counts of ratings per user:
df.stat.crosstab("user","Rating").show()
# average rating given by each user
print(df.groupBy("user").agg({'Rating': 'mean'}).take(5))
# average rating per movie
print(df.groupBy("product").agg({'Rating': 'mean'}).take(5))
# build the engine
(training, test) = ratings.randomSplit([0.8, 0.2])
print(training.count())
print(test.count())
rank = 10
numIterations = 10
model = ALS.train(training, rank, numIterations)
print(model)
# convert testdata to pipelineRDD
testdata = test.map(lambda p: (p[0], p[1]))
print(testdata)
print(test.take(5))
print(testdata.take(5))
# predict user 119 to movie 392
pred_index = model.predict(119, 392)
print(pred_index)
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]),r[2]))
print(type(predictions))
print(predictions.take(5))
# recommend top n items to user using recommendProductsForUsers(n) where n number of items to be recommended
recommendItemsToUsers = model.recommendProductsForUsers(10)
print(recommendItemsToUsers.count())
print(recommendItemsToUsers.take(2))
# evalute the model using mean sqaured root error
ratesAndPreds = ratings.map(lambda r: ( (r[0], r[1]),r[2]) ).join(predictions)
MSE = ratesAndPreds.map(lambda r: ( r[1][0] - r[1][1]) ** 2).mean()
print(MSE)
from math import sqrt
rmse = sqrt(MSE)
print(rmse)
