import numpy as np
from evolutionary_search import maximize

def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
args = {'m': 1.}
best_params, best_score, score_results, _, _ = maximize(func, param_grid, args, verbose=False)

print("best_params = %s" %best_params)
print("best_score = %.2f" %best_score)
print(score_results)