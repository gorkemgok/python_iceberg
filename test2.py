#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:40:41 2017

@author: gorkem
"""
import json
from pprint import pprint
import numpy as npy
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans

with open('train.json') as data_file:    
    data = json.load(data_file)

clData = [];
clMarker = [];
for i in range(0, len(data)):
    M = npy.reshape(data[i]['band_1'], (75,75))
    k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means.fit(M)
    k_means_2 = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means_2.fit(npy.transpose(M))
    [npy.sum(k_means.labels_), npy.sum(k_means_2.labels_), data[0]['is_iceberg']]
    clData.append([npy.sum(k_means.labels_), npy.sum(k_means_2.labels_)])
    clMarker.append('o' if data[i]['is_iceberg'] == 1 else 'x')
