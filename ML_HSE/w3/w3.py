# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 02:08:41 2016

@author: Nikolay
"""

import pandas as pd
import numpy as np
from sklearn import svm

cols=['target','feat1','feat2']
df = pd.read_csv("svm-data.csv", names=cols)

clf = svm.SVC(C=100000, kernel='linear', random_state=241)
clf.fit(df[['feat1', 'feat2']], df['target'])
print(clf.support_)