# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:34:04 2016

@author: Nikolay
"""


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn import datasets

df = datasets.load_boston()
X = scale(df.data)
y = df.target

space = np.linspace(1,10,200)
kf = KFold(len(X),n_folds=5,shuffle=True,random_state=42)
numNeigh = []
for i in space:
    neigh = KNeighborsRegressor(n_neighbors=5,weights='distance',p=i)
    score = np.mean(cross_val_score(neigh,X,y,cv=kf,scoring='mean_squared_error'))
    numNeigh.append([i,score])

