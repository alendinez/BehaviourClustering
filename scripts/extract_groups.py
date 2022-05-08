import sys
sys.path.insert(0,"..") ## Set path to main directory

import os
import time
import numpy as np
import multiprocessing
import csv
import datetime
import pandas as pd
import copy
import more_itertools as mit
import random
from scipy import signal
from functools import partial
from tslearn.clustering import KShape, TimeSeriesKMeans

import models.data as dt
import models.data_manager as data_manager
import models.segment_manager as segment_manager
import models.segment as sgmnt

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_manager and segment_manager    
sigma = 6
w = 100
mode = "mean"
segment_manager = segment_manager(sigma, w, mode)
data_manager = data_manager()

path = "../../Data/output/"
all_data = np.load(path + "all_data.npy", allow_pickle = True)
print("Data loaded")

### Load previously created acceleration segments
all_segments = data_manager.load_all_segments_linux(path, sigma, w)
for data in all_data:
    for segment in all_segments:
        if segment.filename == data.filename:
            segment.setup_acceleration(data)
print("Acceleration data set")

### Load correlation data
maxcorr_ax = np.load(path + "maxcorr_ax.npy")
maxcorr_ay = np.load(path + "maxcorr_ay.npy")
maxcorr_az = np.load(path + "maxcorr_az.npy") 

np.fill_diagonal(maxcorr_ax, 0.0)
np.fill_diagonal(maxcorr_ay, 0.0)
np.fill_diagonal(maxcorr_az, 0.0)

maxcorr = np.dstack((maxcorr_ax, maxcorr_ay, maxcorr_az))
print("Max correlation matrices loaded")

### Segments filtering
segments_out = [idx for idx in range(len(maxcorr)) if np.all(maxcorr[idx].sum(axis=1)<1.8)]
segments_out.sort(reverse=True)
for idx in segments_out:
    all_segments.pop(idx)
print("Segments filtered")

### Delete unnecessary variables
del all_data
del maxcorr
del maxcorr_ax
del maxcorr_ay
del maxcorr_az
del segments_out

### Load model
km_sdtw = TimeSeriesKMeans.from_hdf5(path + 'trained_model_2.hdf5')

labels = km_sdtw.labels_ 
n_clusters = len(km_sdtw.cluster_centers_)
assert len(labels) == len(all_segments)

groups_raw = [[] for i in range(n_clusters)]

for label, segment in zip(labels, all_segments):
    segment.group_label = label
    groups_raw[label].append(segment)

print("Number of raw groups: "+str(len(groups_raw)))
groups_raw = np.array(groups_raw, dtype="object")
np.save(os.path.join(path, 'groups_raw.npy'), groups_raw)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")