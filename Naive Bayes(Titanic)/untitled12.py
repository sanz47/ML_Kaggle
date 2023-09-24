#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 02:20:44 2022

@author: sanz
"""

import pandas as pd
import numpy as np
df = pd.read_csv("train.csv")
p_test=pd.read_csv("test.csv")
X_test=pd.read_csv("test.csv")


X_test.drop(['Name','Ticket','Cabin','Age','Fare'],axis='columns',inplace=True)
df.drop(['Name','Ticket','Cabin','Age','Fare'],axis='columns',inplace=True)

dataset=[df,X_test]

for data in dataset:
    data.Embarked=data.Embarked.fillna('S')
    

new_gen=pd.get_dummies(df.Sex)
df=pd.concat([df,new_gen],axis='columns')
new_gen=pd.get_dummies(df.Embarked)
df=pd.concat([df,new_gen],axis='columns')


ew_gen=pd.get_dummies(X_test.Sex)
X_test=pd.concat([X_test,ew_gen],axis='columns')
new_gen=pd.get_dummies(X_test.Embarked)
X_test=pd.concat([X_test,new_gen],axis='columns')

df.drop(["Embarked","Sex","S","male"],axis="columns",inplace=True)
X_test.drop(["Embarked","Sex","S","male"],axis="columns",inplace=True)

y_test=df['Survived']
df.drop(['Survived','PassengerId'],axis='columns',inplace=True)
X_test.drop(['PassengerId'],axis='columns',inplace=True)

from sklearn.linear_model import LogisticRegression
vv=LogisticRegression(random_state=0)
vv.fit(df, y_test)

y_pred=vv.predict(X_test)

output=pd.DataFrame({'PassengerId':p_test.PassengerId,'Survived':y_pred})
output.to_csv("My_SUB.csv",index=False)