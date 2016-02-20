# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 07:02:30 2016

@author: Nikolay
"""

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_csv("close_prices.csv")
df1 = pd.read_csv("djia_index.csv")
pca = PCA(n_components=10)
pca.fit(df[df.columns[1:]])
X_new = pca.transform(df[df.columns[1:]])

