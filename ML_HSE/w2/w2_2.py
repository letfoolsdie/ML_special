# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:42:18 2016

@author: Nikolay
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

cols=['target','feat1','feat2']
train = pd.read_csv("perceptron-train.csv" ,names=cols)
test = pd.read_csv('perceptron-test.csv', names=cols)

perc = Perceptron(random_state=241)
perc.fit(train[['feat1', 'feat2']],train['target'])
test['pred'] = perc.predict(test[['feat1', 'feat2']])
print(accuracy_score(test['target'], test['pred']))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[['feat1', 'feat2']])
y_scaled = scaler.transform(test[['feat1', 'feat2']])
perc.fit(X_scaled,train['target'])
test['pred_scaled'] = perc.predict(y_scaled)
print(accuracy_score(test['target'], test['pred_scaled']))
