# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:00:47 2016

@author: Nikolay
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

w1 = 0
w2 = 0
k= 0.1
C = 10
cols = ['y','x1','x2']
df = pd.read_csv('data-logistic.csv', names=cols)

i = 0
nor_prev = 0
while True:
    w1_n = w1 + k*np.average(df['y']*df['x1']*(1 - (1/(1+np.exp(-1*df['y']*(w1*df['x1']+w2*df['x2'])))))) - k*C*w1
    w2_n = w2 + k*np.average(df['y']*df['x2']*(1 - (1/(1+np.exp(-1*df['y']*(w1*df['x1']+w2*df['x2'])))))) - k*C*w2
    nor_now = (np.linalg.norm([w1_n,w2_n]))**2
    if (abs(nor_now - nor_prev)) < 10**(-5):
        print('Found it!')
        break
    else:
        nor_prev = nor_now
        w1 = w1_n
        w2 = w2_n
        i+=1
    if i>10000:
        print('Something went wrong :(')
        break

df['alg'] = 1/(1+np.exp(-w1*df['x1']-w2*df['x2']))
print(roc_auc_score(df['y'],df['alg']))

