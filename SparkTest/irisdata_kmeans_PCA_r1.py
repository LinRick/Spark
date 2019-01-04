# coding: utf-8

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.feature import PCA as PCAmllib
from pyspark.mllib.linalg import Vectors
from numpy import array
import numpy as np
from pyspark import SparkContext
from math import sqrt

#from pyspark import SparkContext, SparkConf
#conf = SparkConf().setAppName(appName).setMaster(master)
#sc = SparkContext(conf=conf)

sc = SparkContext("local", "kmeans pca App")

#Load and parse the data
data = sc.textFile("iris_data.txt") #for master local or standalone model

#data = sc.textFile("hdfs://master:9000/root/pyspark_test/iris_data.txt") #for hadoop yarn
#data filter 
parsedData = data.map(lambda line: array([x for x in line.split(',')]))

first_data=parsedData.take(1)[0]
data_row=len(first_data) #include many input and one output attributes

params_only=parsedData.map(lambda x: Vectors.dense(np.float_(x[0:(data_row-1)])))
#params_only.take(5)
#the type of params_only is pyspark.rdd.PipelinedRDD
#params_only=parsedData.map(lambda x: array(np.float_(x[0:(data_row-1)])))


model_test = PCAmllib(2).fit(params_only)
transformed = model_test.transform(params_only)
#transformed.collect()

pca_2d=transformed.collect()

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# kmeans, k=3
k=3
clusters = KMeans.train(params_only, k, maxIterations=10,runs=10, initializationMode="kmeans||") #kmeans|| is kmeans++
#clusters.clusterCenters 各群之群心 list --collect()
#clusters.predict 各資料所屬之群心 list  --collect()
array_centroid=clusters.clusterCenters
rdd_prediction=clusters.predict(params_only)    
WSSSE = params_only.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("With " +str(k)+ " clusters: Within Set Sum of Squared Error = " + str(WSSSE))

data_prediction=rdd_prediction.collect()

data_prediction=array(data_prediction)

# pca後的二維來畫圖
pca_2d=array(pca_2d)


# import pylab
# pylab.figure('K-means with'+str(k)+'clusters')
# ndx = pylab.where(data_prediction == 0)[0]
# pylab.plot(pca_2d[ndx, 0], pca_2d[ndx, 1], 'bo',label='cluster 0')
# ndx = pylab.where(data_prediction == 1)[0]
# pylab.plot(pca_2d[ndx, 0], pca_2d[ndx, 1], 'ro',label='cluster 1')
# ndx = pylab.where(data_prediction == 2)[0]
# pylab.plot(pca_2d[ndx, 0], pca_2d[ndx, 1], 'wo',label='cluster 2')
# pylab.legend()
# pylab.axis('off')
# pylab.savefig('threecluster.png')
# pylab.show()


