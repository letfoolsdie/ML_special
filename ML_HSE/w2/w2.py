# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 01:08:55 2016

@author: Nikolay/letfoolsdie

Data taken from https://archive.ics.uci.edu/ml/datasets/Wine
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
matplotlib.style.use('ggplot')

cols = """
     0) Class
 	1) Alcohol
 	2) Malic acid
 	3) Ash
	4) Alcalinity of ash  
 	5) Magnesium
	6) Total phenols
 	7) Flavanoids
 	8) Nonflavanoid phenols
 	9) Proanthocyanins
	10) Color intensity
 	11) Hue
 	12) OD280/OD315 of diluted wines
 	13) Proline   
  """

cols = cols.split(') ')
cols = [i[:-3] for i in cols]
cols = [i.replace('\n','').strip() for i in cols]
cols = cols[1:]

df = pd.read_csv('wine.csv', names = cols)
X = df[cols[1:]]
y = df[cols[0]]
X = scale(X) #comment this line out for not preprocessing (scaling) features
kf = KFold(len(X),n_folds=5,shuffle=True,random_state=42)
numNeigh = []
for i in range(50):
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    score = np.mean(cross_val_score(neigh,X,y,cv=kf,scoring='accuracy'))
    numNeigh.append([i+1,score])
numNeigh = np.array(numNeigh)
plt.plot(numNeigh[:,0],numNeigh[:,1])


