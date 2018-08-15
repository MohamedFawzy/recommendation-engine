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
