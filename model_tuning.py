# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:56:03 2021

@author: rayde
"""


from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

#Grid search
params = {'n_neighbors': np.arange(1,3), 'metric': ['euclidean', 'cityblock']}
grid = GridSearchCV(estimator=knn, param_grid=params)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

#Randomized Parameter Optimization

params = {'n_neighbors': range(1,5), 
          'weights': ['uniform', 'distance']}

rsearch = RandomizedSearchCV(estimator=knn, 
                             param_distributions = params, 
                             cv=4, 
                             n_iter=8,
                             random_state=5)

rsearch.fit(X_train, y_train)
print(rsearch.best_score_)
