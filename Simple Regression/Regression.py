#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun 24 17:04:27 2022

@author: sanz

Linear Regression 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


#Fitting Regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#predict
y_pred=regressor.predict([[65]])

#visualize Test sets
print((y_pred))