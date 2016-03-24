# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 01:22:54 2016

@author: Nikolay
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

###f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)

f = lambda x: np.sin(x/5)*np.exp(x/10) + 5 * np.exp(-x/2)
#b = np.array([f(1),f(15)])
#A = np.array([[1,1],[1,15]])
#ans = np.linalg.solve(A,b)

###Для многочлена 3-й степени:
power = 3
points = [1,4,10,15]
b = np.array([f(i) for i in points])
A = np.array([[coef**i for i in range(power+1)] for coef in points])
ans = np.linalg.solve(A,b)


###Visualization
##Print the f function:
f2 = lambda x,w: w[0]+w[1]*x+w[2]*x**2+w[3]*x**3
x = np.arange(1,15,0.1)
y = f2(x,ans)
#y = f(x)
#plt.plot(x,y)
#
#p3 = interpolate.interp1d(points, b, kind=3)
#xnew = np.arange(1,15,0.1)
#ynew = p3(xnew)
#plt.plot(xnew,ynew)
