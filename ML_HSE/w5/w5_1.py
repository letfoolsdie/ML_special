# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:05:56 2016

@author: Nikolay
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
 
plt.style.use('ggplot')

df = pd.read_csv('gbm-data.csv')
val = df.values
X = val[:,1:]
y = val[:,0]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.8, random_state=241)

#learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
learning_rates  = [0.2]
sigmoid = lambda x: 1 / (1 + np.exp(-x))

log_loss_test = []
for l in learning_rates:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True,
                                     random_state=241,learning_rate = l)
    print('fitting...')
    clf.fit(X_train, y_train)
    print('building staged_decision_function')
    staged_dec = clf.staged_decision_function(X_test)
    for pred in staged_dec:
        y_pred = sigmoid(pred)
        log_loss_test.append(log_loss(y_test,y_pred))
best_iter = [np.argmin(log_loss_test),log_loss_test[np.argmin(log_loss_test)]]
#clf1 = RandomForestClassifier(n_estimators = 37, random_state=241)
#clf1.fit(X_train, y_train)
#prediction = clf1.predict_proba(X_test)
#res = log_loss(y_test,prediction)
#        