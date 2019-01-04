#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Load and parse the data

from pyspark.mllib.regression import LabeledPoint

def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])