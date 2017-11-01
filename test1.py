# -*- coding: utf-8 -*-

import json
from pprint import pprint
import numpy as npy
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans

with open('train.json') as data_file:    
    data = json.load(data_file)

M = npy.reshape(data[0]['band_1'], (75,75))
k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
k_means.fit(M)
k_means_2 = KMeans(init='k-means++', n_clusters=2, n_init=10)
k_means_2.fit(npy.transpose(M))
plot.figure(1);
plot.subplot(221)
plot.imshow(npy.reshape(k_means_2.labels_, (1,75) ));
plot.subplot(223)
plot.imshow(M, cmap='hot', interpolation='nearest');
plot.subplot(224)
plot.imshow(npy.reshape(k_means.labels_, (75,1) ));
plot.show();