# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 03:03:25 2016

@author: Nikolay
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn import grid_search

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

vect = TfidfVectorizer()
X = vect.fit_transform(newsgroups.data)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(len(newsgroups.data), n_folds=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, newsgroups.target)
#print(gs.grid_scores_)
C = 1.0 #C with best error (min C, if several C with equal values of error)
clf.fit(X,newsgroups.target) #C already equal 1.0 (by default)

resNumber=np.argsort(np.absolute(np.asarray(clf.coef_.todense()).reshape(-1)))[-10:]
words = []
for i in resNumber:
    words.append(vect.get_feature_names()[i])
    

#words - top 10 words with maximum absolute weight