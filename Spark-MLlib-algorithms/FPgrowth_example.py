#!/usr/bin/python
# -*- coding: UTF-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.mllib.fpm import FPGrowth

if __name__ == "__main__":
  sc   = SparkContext()
  data = sc.textFile("/root/spark-2.3.1-bin-hadoop2.6/data/mllib/sample_fpgrowth.txt")
  transactions = data.map(lambda line: line.strip().split(' '))
  model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
  result = model.freqItemsets().collect()
  for fi in result:
    print(fi)
  sc.stop()
