# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:39:56 2016

@author: Nikolay
"""

import pandas
import numpy as np
#import time
#import datetime
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import grid_search


def bagOfWords(X):
    N = 112
    X_pick = np.zeros((X.shape[0], N))    
    for i, match_id in enumerate(X.index):
        for p in xrange(5):
            X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    return X_pick


def getCategorColumns(X):
    categor_cols =  []
    for c in X.columns:
        if c.find('_hero')>0:
            categor_cols.append(c)
    categor_cols.append('lobby_type')
    return categor_cols
    

def bestC(clf, X, y):
    #Функция для поиска оптимального С
    kf = KFold(len(X), n_folds=5, shuffle=True, random_state=seed)
    grid = {'C': np.power(10.0, np.arange(-5, 5))}
    gs = grid_search.GridSearchCV(clf, grid, scoring='roc_auc', cv=kf)
    gs.fit(X, y)
    return gs.best_estimator_, gs.grid_scores_
    
seed = 42
features = pandas.read_csv('./data/features.csv', index_col='match_id')
X = features.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_dire',\
                   'barracks_status_radiant'],axis=1)                   
y = features['radiant_win']
categor_cols = getCategorColumns(X)

X = X.drop(categor_cols, axis=1)



X.fillna(400, inplace=True)
#X_scaled = StandardScaler().fit_transform(X)
#X_bag = bagOfWords(features)
X_train = np.hstack((StandardScaler().fit_transform(X), bagOfWords(features)))

#clfLog, gsscores = bestC(LogisticRegression(random_state=seed), X_train, y)
kf = KFold(len(X), n_folds=5, shuffle=True, random_state=seed)
clfLog = LogisticRegression(C=0.10000000000000001, random_state=seed)
scores = cross_val_score(clfLog, X_train, y, scoring='roc_auc', cv=kf)

