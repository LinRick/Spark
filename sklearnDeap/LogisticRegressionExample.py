
import sklearn.datasets
import numpy as np
import random
import pandas as pd

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

paramgrid = {"tol": np.logspace(-2, -1, num=2, base=10),
             "max_iter" : np.logspace(1, 1, num=2, base=10)}


random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
"""
cv = EvolutionaryAlgorithmSearchCV(estimator=LogisticRegression(),
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
"""
                                   
cv = EvolutionaryAlgorithmSearchCV(estimator=LogisticRegression(),
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

print("cv.best_score_=%s" %cv.best_score_)
print("cv.best_params_=%s" %cv.best_params_)

print(pd.DataFrame(cv.cv_results_).sort_values("mean_test_score", ascending=False))