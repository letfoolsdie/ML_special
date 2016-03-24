# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:53:32 2016

@author: Nikolay
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

f = lambda x: np.sin(x/5)*np.exp(x/10) + 5 * np.exp(-x/2)
x = np.arange(1,30,0.1)
y = np.array(f(x))
ans = minimize(f,30,method='BFGS') ##minimizing using gradient
#ans = differential_evolution(f,[(1,30)])

###Minimize not-smooth (не гладкую) function
#x = np.arange(1,30,0.1)
#h = lambda x: (np.sin(x/5)*np.exp(x/10) + 5 * np.exp(-x/2)).astype(int)
#y = np.array(h(x))
##ans = minimize(h,30,method='BFGS') ##minimizing using gradient (which 
###                                           sucks for this function)
#ans1 = differential_evolution(h,[(1,30)])
#print (ans1)
#print(ans1)
plt.plot(x,y)