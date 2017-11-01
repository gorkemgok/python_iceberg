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
from sklearn.metrics import log_loss

"""with open('octave_workspace/kaggle/iceberg/train.json') as data_file:"""
with open('train.json') as data_file:    
    data = json.load(data_file)
    
dArray = []
for i in range(0, len(data)):
    m = npy.reshape(data[i]['band_1'], (75,75))
    k_means_y_band1 = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means_y_band1.fit(m)
    k_means_x_band1 = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means_x_band1.fit(npy.transpose(m))
    m = npy.reshape(data[i]['band_2'], (75,75))
    k_means_y_band2 = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means_y_band2.fit(m)
    k_means_x_band2 = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means_x_band2.fit(npy.transpose(m))
    
    dArray.append([data[i]['band_1'], data[i]['band_2'], data[i]['is_iceberg'],
                   npy.sum(k_means_y_band1.labels_), npy.sum(k_means_x_band1.labels_),
                   npy.sum(k_means_y_band2.labels_), npy.sum(k_means_x_band2.labels_)])
    
D = npy.array(dArray)
kMeans_band_1 = KMeans(init='k-means++', n_clusters=2, n_init=10)
kMeans_band_1.fit(D[:,0].tolist());

D = npy.array(dArray)
kMeans_band_2 = KMeans(init='k-means++', n_clusters=2, n_init=10)
kMeans_band_2.fit(D[:,1].tolist());

features = []
labels = []
for i in range(0, len(data)):
    features.append([kMeans_band_1.labels_[i], kMeans_band_2.labels_[i], 
                     min([D[i][3], D[i][4]]), max([D[i][3], D[i][4]]),
                     min([D[i][5], D[i][6]]), max([D[i][5], D[i][6]])]);
    labels.append(D[i][2])
    
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(features, labels)

predictions = []
for i in range(0, len(features)):
    predictions.append(clf.predict([features[i]]))

log_loss(labels, predictions)

tree.export_graphviz(clf)
"""
wrong = 0;
for i in range(0, len(labels)):
    wrong += 1
    pprint(str(labels[i]) + ' - ' + str(predictions[i]))
    if labels[i] != predictions[i]:
        wrong += 1
"""