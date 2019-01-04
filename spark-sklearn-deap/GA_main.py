#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Created on 2018年9月10日

@author: A40404
'''

import sklearn.datasets
import numpy as np
import random
from sklearn.svm import SVC

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from evolutionary_search import maximize


from sklearn.model_selection import StratifiedKFold

import random
from deap import creator, base, tools, algorithms

if __name__ == '__main__':

    data = sklearn.datasets.load_digits()
    
    X = data["data"]
    y = data["target"]
    
    paramgrid = {"kernel": ["rbf"],
                 "C"     : np.logspace(-1, 1, num=25, base=10),
                 "gamma" : np.logspace(-1, 1, num=25, base=10)}
    random.seed(1)
    
    cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                       params=paramgrid,
                                       scoring="accuracy",
                                       cv=StratifiedKFold(n_splits=4),
                                       verbose=1,
                                       population_size=50,
                                       gene_mutation_prob=0.10,
                                       gene_crossover_prob=0.5,
                                       tournament_size=3,
                                       generations_number=5,
                                       n_jobs=4)
    cv.fit(X, y)






