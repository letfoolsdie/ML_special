# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:00:40 2016

@author: Nikolay
"""

from sklearn.datasets import load_digits
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


import pandas as pd
import numpy as np

def writeans(ans, filename):
    with open(filename + '.txt', 'w') as f:
        f.write(str(ans))
        f.close()
        

digits = load_digits()

X = digits.data
y = digits.target

clf = DecisionTreeClassifier()
kf = KFold(len(X), n_folds=10)
#scores = cross_val_score(clf, X, y, cv=kf)
#print(scores.mean())
#writeans(scores.mean(), '1')
#
###################
#
#clfbag = BaggingClassifier(n_estimators=100)
#scores_bag = cross_val_score(clfbag, X, y, cv=kf)
#print(scores_bag.mean())
#writeans(scores_bag.mean(), '2')
#
###################
#
#clfbag_n = BaggingClassifier(n_estimators=100, max_features=8) ##8 - sqrt(features)
#scores_bag_n = cross_val_score(clfbag_n, X, y, cv=kf)
#print(scores_bag_n.mean())
#writeans(scores_bag_n.mean(), '3')

#################

#clfbag_nt = BaggingClassifier(base_estimator= \
# DecisionTreeClassifier(max_features=8), n_estimators=100)
#scores_bag_nt = cross_val_score(clfbag_nt, X, y, cv=kf)
#print(scores_bag_nt.mean())
#writeans(scores_bag_nt.mean(), '4')

##################
clfrf = RandomForestClassifier(n_estimators=100)
scores_bag_rf = cross_val_score(clfrf, X, y, cv=kf)
print(scores_bag_rf.mean())

    