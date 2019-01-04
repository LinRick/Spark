#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import datetime


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

if __name__ == "__main__":
    
    #count = 0
    #aList = []
    #while count<10: 
            
    with open('/root/spark-2.3.1-bin-hadoop2.6/data/mllib/ridge-data/very-big-lpsa.data','r') as f:
      #for line in f:
      readData=f.read().replace(',', ' ').replace('\n', ' ').strip().split(' ')
      #readData=f.read().replace(',', ' ').split(' ')
      
      
    #  aList.extend(readData)
    #  count+=1
      
    starttime = datetime.datetime.now()
    #readData=aList

    
    # 開啟檔案
    #fp = open('/root/spark-2.3.1-bin-hadoop2.6/data/mllib/ridge-data/very-big-lpsa.data', "")
    #fp.writelines(readData)
    
    
        
    floatReadData = [float(x) for x in readData]
    nparrayReadData = np.asarray(floatReadData)
    X = np.delete(nparrayReadData.reshape(nparrayReadData.size/9,9), 0, axis=1)
    Y = (nparrayReadData[0::9])
    clf = linear_model.SGDRegressor(max_iter = 100,eta0 = 0.00000001, 
                                    fit_intercept=False,penalty='none',
                                    epsilon=0.01,l1_ratio=1,learning_rate='invscaling') 
    #for pyspark at iterations=100, step=0.00000001
    model=clf.partial_fit(X,Y)
    y_predicted = model.predict(X)
    mse_val = mse(y_predicted, Y)
    
    endtime = datetime.datetime.now()
    
    print (X.shape)
    print (starttime)
    print (endtime)
    print ("Running time = %d seconds") %(endtime - starttime).seconds
    
    print("Mean Squared Error = " + str(mse_val))
        
    


