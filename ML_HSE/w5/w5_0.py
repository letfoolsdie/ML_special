# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 06:39:06 2016

@author: Nikolay
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np

df = pd.read_csv('abalone.csv')
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
features = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight',
       'VisceraWeight', 'ShellWeight']
target = ['Rings']
X = df[features]
y = df.Rings

kf = KFold(len(df),n_folds=5, shuffle=True, random_state=1)
for t in range(1,51):
    clf = RandomForestRegressor(n_estimators=t, random_state=1)
    score = cross_val_score(clf, X, y, cv=kf, scoring='r2')
    print([t,np.mean(score)])