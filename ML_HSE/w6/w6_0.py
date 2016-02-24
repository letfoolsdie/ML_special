# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:10:24 2016

@author: Nikolay
"""

from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

image = imread('parrots.jpg')
image = img_as_float(image)

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image.shape)
image_array = np.reshape(image, (w * h, d))

results = []
for i in range(8,20):
    clf = KMeans(n_clusters = i, init='k-means++', random_state = 241)
    clf.fit(image_array)
    
    reduced = np.array([clf.cluster_centers_[i] for i in clf.labels_])
    mse = mean_squared_error(image_array,reduced)
    psnr = -10*np.log10(mse)
    results.append([i,psnr])
    if psnr>20:
        break
