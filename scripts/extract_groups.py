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
import models.KShapeVariableLength as KShapeVariableLength

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_manager and segment_manager    
sigma = 0.3
w = 150
mode = "std"
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

### Segments filtering
segments_copy = all_segments.copy()
segments_copy.sort(key=lambda x: len(x.ax), reverse=True)
l = len(segments_copy)
segments_out = [x.id for x in segments_copy[:int(l*0.05)]]
segments_out.sort(reverse=True)
for idx in segments_out:
    all_segments.pop(idx)
print("Segments filtered")

i = 0
for segment in all_segments:
    segment.id = i
    i = i+1
print("Segments reindexed")

### Load model
ksvl = KShapeVariableLength.from_hdf5(path + 'ksvl_16_clusters.hdf5')

labels = ksvl.labels_ 
n_clusters = len(ksvl.cluster_centers_)
assert len(labels) == len(all_segments)

groups_raw = [[] for i in range(n_clusters)]

for label, segment in zip(labels, all_segments):
    #segment.group_label = label
    groups_raw[label].append(segment.id)

print("Number of raw groups: "+str(len(groups_raw)))
groups_raw = np.array(groups_raw, dtype="object")
np.save(os.path.join(path, 'groups_raw_16.npy'), groups_raw)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")