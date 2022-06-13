import sys
sys.path.insert(0,"..") ## Set path to main directory

import os
import time
import numpy as np
import datetime
import copy
import random
from scipy import signal
from functools import partial
import seaborn as sns
from matplotlib import pyplot as plt

import models.data as dt
import models.data_manager as data_manager
import models.segment_manager as segment_manager
import models.segment as sgmnt
import models.KShapeVariableLength as KShapeVariableLength

from tslearn.clustering.utils import (TimeSeriesCentroidBasedClusteringMixin,
                    _check_no_empty_cluster, _compute_inertia,
                    _check_initial_guess, EmptyClusterError)

import warnings
warnings.filterwarnings("ignore")

def compute_inertia(cross_dists, labels, cluster_centers, n_clusters):
    dists = 1. - cross_dists
    _check_no_empty_cluster(labels, n_clusters)
    return _compute_inertia(dists[:,cluster_centers], labels)

def centroid_selection(cross_dists, labels, k):
    # Select submatrix of cross dists with label k
    idxs = (labels == k).nonzero()[0]
    dists = 1. - cross_dists[np.ix_(idxs,idxs)]
    # Get the one with max value
    centroid = idxs[dists.sum(axis=1).argmin()]
    return centroid

start_time = time.time()

sigma = 0.3
w = 150
mode = "std"

data_manager = data_manager()

path = "../../Data/output/"

### Load all data
all_data = np.load(path + "all_data.npy", allow_pickle = True)
print("Data loaded")

### Load previously created acceleration segments
all_segments = data_manager.load_all_segments_linux(path, sigma, w)
for data in all_data:
    for segment in all_segments:
        if segment.filename == data.filename:
            segment.setup_acceleration(data)
print("Acceleration data set")

### Segments filtering
all_segments.sort(key=lambda x: len(x.ax), reverse=True)
l = len(all_segments)
segments_in = [x.id for x in all_segments[int(l*0.05):]]
segments_in.sort()
assert segments_in[0] == min(segments_in)
print("Segments filtered")

maxcorr = np.load(path + f"maxcorr_{sigma}_{w}.npy")
maxcorr = maxcorr[np.ix_(segments_in, segments_in)]
print("Max correlation matrix loaded:", maxcorr.shape)

groups = np.load(path + "groups_raw_24_4.npy", allow_pickle = True)

### Get labels array
labels = np.empty(len(maxcorr), dtype=int)
for i in range(len(groups)):
    for idx in groups[i]:
        labels[idx] = i

### Get centroids
n_clusters = len(groups)
cluster_centers = np.empty(n_clusters, dtype=int)
for i in range(n_clusters):
    cluster_centers[i] = centroid_selection(maxcorr, labels, i)

print("Cluster centers:", cluster_centers)
inertia = compute_inertia(maxcorr, labels, cluster_centers, n_clusters)
print("Inertia:", inertia)

for k in range(n_clusters):
    idxs = (labels == k).nonzero()[0]
    dists = 1. - maxcorr[np.ix_(idxs,[cluster_centers[k]])]
    
    sns.distplot(dists, hist=True, kde=True, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.title(f'Group {k}')
    plt.show()

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")