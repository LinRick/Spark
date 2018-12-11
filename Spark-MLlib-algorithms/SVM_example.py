#!/usr/bin/python
# -*- coding: UTF-8 -*-
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

if __name__ == "__main__":
  sc   = SparkContext()  
  data = sc.textFile("/root/spark-2.3.1-bin-hadoop2.6/data/mllib/sample_svm_data.txt")
  parsedData = data.map(parsePoint)

  # Build the model
  model = SVMWithSGD.train(parsedData, iterations=100)

  # Evaluating the model on training data
  labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())

  print("Training Accuracy = " + str(1-trainErr))
  print("Training Error = " + str(trainErr))

  # Save and load model
  #model.save(sc, "target/tmp/pythonSVMWithSGDModel")
  #sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")
  sc.stop()
