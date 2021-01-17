# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:25:46 2021

@author: rayde
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#UNSUPERVISED LEARNING ESTIMATORS

#Principal Component Analysis (PCA)
pca = PCA(n_components=.95)

#K Means model fitting
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)

#Prediction for clustering algos
y_pred = k_means.predict(X_test)
