#!/usr/bin/python
# -*- coding: UTF-8 -*-
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.regression import LassoWithSGD, LassoModel
from pyspark.mllib.regression import RidgeRegressionWithSGD, RidgeRegressionModel

from pyspark import SparkConf, SparkContext

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

if __name__ == "__main__":
  sc   = SparkContext()  
  data = sc.textFile("/root/spark-2.3.1-bin-hadoop2.6/data/mllib/ridge-data/lpsa.data")
  parsedData = data.map(parsePoint)

  # Build the model
  model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001) # Linear least squares
  #model = LassoWithSGD.train(parsedData, iterations=100, step=0.00000001)           # Lasso
  #model = RidgeRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001) # Ridge regression


  # Evaluate the model on training data
  valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
  MSE = valuesAndPreds \
      .map(lambda vp: (vp[0] - vp[1])**2) \
      .reduce(lambda x, y: x + y) / valuesAndPreds.count()
  print("Mean Squared Error = " + str(MSE))

  # Save and load model
  #model.save(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
  #sameModel = LinearRegressionModel.load(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
  sc.stop()