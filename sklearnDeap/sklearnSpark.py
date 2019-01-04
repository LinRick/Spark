#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn import svm, grid_search, datasets
from spark_sklearn import GridSearchCV
from pyspark import SparkConf, SparkContext

if __name__ == "__main__":
  sc   = SparkContext()
  iris = datasets.load_iris()
  parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
  svr = svm.SVC()
  clf = GridSearchCV(sc, svr, parameters)
  clf.fit(iris.data, iris.target)
  print(clf.cv_results_)
