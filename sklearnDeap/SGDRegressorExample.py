
import sklearn.datasets
import numpy as np
import random

data = sklearn.datasets.load_digits()
n_samples, n_features = 10, 5
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import StratifiedKFold

paramgrid = {"penalty": ["l2"],
             "tol": [0.001],
             "max_iter" : np.logspace(1, 2, num=5, base=10)}


random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV

cv = EvolutionaryAlgorithmSearchCV(estimator=SGDRegressor(),
                                   params=paramgrid,
                                   scoring="neg_mean_squared_log_error",
                                   cv=StratifiedKFold(n_splits=2),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)
cv.fit(X, y)
cv.best_score_, cv.best_params_

pd.DataFrameDataFra (cv.cv_results_).sort_values("mean_test_score", ascending=False).head()

