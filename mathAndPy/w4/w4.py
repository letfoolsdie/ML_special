# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 01:00:11 2016

@author: Nikolay
"""

from scipy.stats import laplace,beta,norm
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

a = 0.4
b = 0.5
#d = laplace()
bdistr = beta(a=a,b=b)
main_sample = bdistr.rvs(1000)
x = np.linspace(0,1,200)
pdf = bdistr.pdf(x)
plt.plot(x,pdf)
plt.hist(main_sample, bins=20, normed=True)
#Для выборок размером 5:
sampleMean5 = np.array([np.mean(np.random.choice(main_sample,5)) for i in\
                                                 range(1000)])
#Для выборок размером 10:
sampleMean10 = np.array([np.mean(np.random.choice(main_sample,10)) for i in\
                                                 range(1000)])    

#Для выборок размером 50:
sampleMean50 = np.array([np.mean(np.random.choice(main_sample,50)) for i in\
                                                 range(1000)])                                                  

#Мат. ожидание Бета-распределения равно a/(a+b)
mu = a/(a+b)                               
#Дисперсия равна (a*b)/((a+b)**2 * (a+b+1)). Т.к. в scipy при задании
# нормального распределения мы указываем среднеквадратчное отклонение, то нам 
#нужен квадратный корень из дисперсии:
sigma = ((a*b)/((a+b)**2 * (a+b+1)))/10
norm_approxim = norm(loc=mu, scale=sigma)       
pdf1 = norm_approxim.pdf(x)
plt.hist(sampleMean10, bins=20, normed=True)
plt.plot(x,pdf1)
                              