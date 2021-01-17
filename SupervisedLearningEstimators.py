# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:21:51 2021

@author: rayde
"""
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

#Supervised learning estimators

#LINEAR REGRESSION
lr = LinearRegression(normalize=True)
print(lr)

#Support vector machines (SVM)
svc = SVC(kernel='linear')
print(svc)

#Naive Bayes
gnb = GaussianNB()
print(gnb)

#KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
print(knn)

#model fitting

lr.fit(X,y)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

#Prediction 
#predict labels
y_pred = svc.predict(np.random.random((2,5)))
#predict labels
y_pred= lr.predict(X_test)
#estimate probability of a label
y_pred = knn.predict_proba(X_test)
