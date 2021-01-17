# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:37:00 2021

@author: rayde
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score
from sklearn.cross_validation import cross_val_score

#PERFORMANCE EVALUATION
#CLASSIFICATION METRICS

#ACCURACY SCORE 
knn.score(X_test, y_test) #estimator score method
accuracy_score(y_test, y_pred)

#Classification Report
print(classification_report(y_test, y_pred)) #precision recall f1 score, and support

#confusion matrix
print(confusion_matrix(y_test, y_pred))

#mean absolute error
y_true = [3, -.5, 2]
mean_absolute_error(y_true, y_pred)

#R2 score
r2_score(y_true, y_pred)

#CLUSTERING METRICS
#Adjusted Rand Index
adjusted_rand_score(y_true, y_pred)

#Homogeneity
homogeneity_score(y_true, y_pred)

#V-measure
metrics.v_measure_score(y_true, y_pred)

#CROSS-VALIDATION
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_val_score(lr, X, y, cv=2))