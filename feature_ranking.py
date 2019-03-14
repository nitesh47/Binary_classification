#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:28:42 2019

@author: nitesh
"""

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Feature_extraction(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = ExtraTreesClassifier()
        self.labelencoder_X_1 = LabelEncoder()
        self.labelencoder_X_2 = LabelEncoder()
        self.onehotencoder = OneHotEncoder(categorical_features = [1])

    ''' Independent feature set'''
    def X(self):
        X = self.dataset.iloc[:, 3:13].values
        return X
    ''' Dependent feature set '''
    def y(self):
        y = self.dataset.iloc[:, 13].values
        return y

    ''' OneHot encoding to categorical values'''
    def Encoding_cat_data(self):
        X = Feature_extraction.X(self) 
        X[:, 1] = self.labelencoder_X_1.fit_transform(X[:, 1])
        X[:, 2] = self.labelencoder_X_2.fit_transform(X[:, 2])
        X = self.onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]
        return X
    ''' Model initialization'''
    def feature_extraction(self):
        return self.model
        
if __name__ == '__main__':
    
    data = pd.read_csv('/home/nitesh/rst/section4/Churn_Modelling.csv')
    
    main = Feature_extraction(data)
    X = main.Encoding_cat_data()
    y = main.y()
    
    model = main.feature_extraction()
    model = model.fit(X,y)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()