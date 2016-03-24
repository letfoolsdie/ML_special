# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 00:03:16 2016

@author: Nikolay
"""
import re
import numpy as np
from scipy.spatial.distance import cosine

f = open('sentences.txt','r')
total = []
for line in f:
    a = line.lower()
    a = re.split('[^a-z]', a)
    a = [i for i in a if i != '']
    total.append(a)
    
#wordCount = {}
#for sent in total:
#    for word in sent:
#        if word in wordCount:
#            wordCount[word] += 1
#        else:
#            wordCount[word] = 0

wordCount = []
for sent in total:
    for word in sent:
        if word in wordCount:
            continue#+= 1
        else:
            wordCount.append(word)



a = np.array([[sent.count(i) for i in wordCount] for sent in total])
k = 0
for i in a:
    print([k,cosine(a[0],i)])
    k+=1
