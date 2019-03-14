#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 19:39:54 2019

@author: nitesh
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import keras
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense



class ANN(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labelencoder_X_1 = LabelEncoder()
        self.labelencoder_X_2 = LabelEncoder()
        self.onehotencoder = OneHotEncoder(categorical_features = [1])
        self.sc = StandardScaler()

        
    def X(self):
        X = self.dataset.iloc[:, 3:13].values
        return X
    
    def y(self):
        y = self.dataset.iloc[:, 13].values
        return y

    def Encoding_cat_data(self):
        X = ANN.X(self) 
        X[:, 1] = self.labelencoder_X_1.fit_transform(X[:, 1])
        X[:, 2] = self.labelencoder_X_2.fit_transform(X[:, 2])
        X = self.onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]
        return X
    
    def standard_scaling(self,X_train,X_test):
        X_train = self.sc.fit_transform(X_train)
        X_test = self.sc.transform(X_test)
        return X_train, X_test
    
    def build_classifier(self):
        classifier = Sequential()
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
        
    def predict(self,X_test):
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        return y_pred
    
if __name__ == '__main__':
    
    corpus = pd.read_csv('/home/nitesh/rst/section4/Churn_Modelling.csv')
    
    Main = ANN(corpus)
    X = Main.Encoding_cat_data()
    y = Main.y()
    
    ''' train/test split '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    ''' feature scaling '''
    X_train,X_test = Main.standard_scaling(X_train,X_test)
    
    ''' Applying PCA to avoid curse of dimensionality problem'''
    pca = PCA(n_components=8)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    ''' Upsampling method----> to balance the classes only on training data
    to avoid the information leakage problem'''
    smt = SMOTE() 
    [X_train, y_train] = smt.fit_resample(X_train,y_train)

    classifier = Main.build_classifier()
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)
    predict = Main.predict(X_test)
