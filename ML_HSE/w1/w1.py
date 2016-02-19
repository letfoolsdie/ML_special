# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:43:24 2016

@author: Nikolay
"""

##Training decision tree on 4 features and calculating feature importances

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('titanic.csv',index_col='PassengerId')

features = ['Pclass','Fare','Age','Sex']
df = df[features+['Survived']]
df.dropna(axis=0,inplace=True)

df.replace(to_replace={'Sex':{'male':0,'female':1}},inplace=True)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(df[features],df['Survived'])

importances = clf.feature_importances_
