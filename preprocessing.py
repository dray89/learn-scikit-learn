# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:09:06 2021

@author: rayde
"""
'''
Your data needs to be numeric and stored as NumPy arrays or SciPy 
sparse matrices. Other types that are convertible to numeric arrays,
such as Pandas DataFrame, are also acceptable.
'''
#LOAD DATA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

X = np.random.random((10 , 5))
y = np.array(['M', 'M', 'F', 'F', 'M', 
              'F', 'M', 'M', 'F', 'F'])
X[X<.7] = 0 

#SPLIT DATA INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_test)
#PREPROCESS THE DATA

#Standardization
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

print(standardized_X_test)

#Normalization
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
print(normalized_X_test)

#Binarization
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
print(binary_X)

#Encoding Categorical Features
enc  = LabelEncoder()
y = enc.fit_transform(y)
#Imputing Missing Values
imp = SimpleImputer(missing_values=0, strategy="mean")
imp.fit_transform(X_train)

#Generating Polynomial Features
poly = PolynomialFeatures(5)
poly.fit_transform(X)