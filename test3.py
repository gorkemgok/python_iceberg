#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:45:53 2017

@author: gorkem
"""

import json
from pprint import pprint
import numpy as npy
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans
from sklearn import tree

"""with open('octave_workspace/kaggle/iceberg/train.json') as data_file:"""
with open('train.json') as data_file:    
    data = json.load(data_file)
    
dArray = []
for i in range(0, len(data)):
    m = npy.reshape(data[i]['band_1'], (75,75))
    k_means_y = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means_y.fit(m)
    k_means_x = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means_x.fit(npy.transpose(m))
    dArray.append([data[i]['band_1'], data[i]['band_2'], data[i]['is_iceberg'],
                   npy.sum(k_means_y.labels_), npy.sum(k_means_x.labels_)])
    
D = npy.array(dArray)
kMeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
kMeans.fit(D[:,0].tolist());

features = []
labels = []
for i in range(0, len(data)):
    features.append([kMeans.labels_[i], D[i][3], D[i][4]]);
    labels.append(D[i][2])
    
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

predictions = []
for i in range(0, len(features)):
    predictions.append([features[i])])