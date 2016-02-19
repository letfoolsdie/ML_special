# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 02:08:41 2016

@author: Nikolay
"""

import pandas as pd
import numpy as np
from sklearn import metrics


###PART 1
#df = pd.read_csv('classification.csv')
#
#TP = sum((df.true == df.pred) & (df.pred == 1))
#FP = sum((df.true != df.pred) & (df.pred == 1))
#FN = sum((df.true != df.pred) & (df.pred == 0))
#TN = sum((df.true == df.pred) & (df.pred == 0))
#
#accuracy = metrics.accuracy_score(df.true,df.pred)
#precision = metrics.precision_score(df.true,df.pred)
#recall = metrics.recall_score(df.true,df.pred)
#f1 = metrics.f1_score(df.true,df.pred)

 
###PART 2
df = pd.read_csv('scores.csv')

for i in df.columns[1:]:
    roc = metrics.roc_auc_score(df.true,df[i])
    print([i,roc])
    
for i in df.columns[1:]:
    precision, recall, thresholds = \
                metrics.precision_recall_curve(df.true,df[i])
    output = max(precision[recall>0.7])
    print([i,output])