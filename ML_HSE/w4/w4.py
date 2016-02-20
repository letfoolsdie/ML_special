# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 03:12:03 2016

@author: Nikolay
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

df = pd.read_csv('salary-train.csv')


fun = lambda x: x.lower()
df['FullDescription'] = df.FullDescription.apply(fun)
df['FullDescription'] = df['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)


vect = TfidfVectorizer(min_df=5)
X = vect.fit_transform(df['FullDescription'])

df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))
train = hstack([X,X_train_categ])

clf = Ridge(alpha=1.0)
clf.fit(train, df.SalaryNormalized)

##Prepare test data:
test = pd.read_csv('salary-test-mini.csv')
test['FullDescription'] = test.FullDescription.apply(fun)
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
Y = vect.transform(test['FullDescription'])
Y_train_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
fintest = hstack([Y,Y_train_categ])
answer = clf.predict(fintest)