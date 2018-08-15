# load  user ratings dataset
# author: MohamedFawzy
# email: mfawzy22@gmail.com
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark import SparkContext
# create spark context app
sc = SparkContext(appName = "UserBase")
data = sc.textFile("/var/www/html/ml-100k/u.data")
print("====================type===================")
print(type(data))
print("============================================")
