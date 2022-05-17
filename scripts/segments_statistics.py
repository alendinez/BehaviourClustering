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
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import models.data as dt
import models.data_manager as data_manager
import models.segment_manager as segment_manager
import models.segment as sgmnt

start_time = time.time()

### Initialize data_manager and segment_manager    
sigma = 0.3
w = 200
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
'''
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
'''
print("Number of segments:", len(all_segments))
print("")

#Get distances between segments
dists = []
prev = 0
sgmnt_min = None
for sgmnt in all_segments:
    if(prev < sgmnt.start):
        dists.append(sgmnt.start - prev)
    prev = sgmnt.end
dists.sort()
l = len(dists)
print("Min distance:", dists[0])
print("5 percent distance:", dists[int(l*0.05)])
print("1st quantile distance:", dists[int(l*0.25)])
print("Median distance:", dists[int(l*0.5)])
print("3rd quantile distance:", dists[int(l*0.75)])
print("Max distance:", dists[-1])
print("")

all_segments.sort(key=lambda x: len(x.ax))
print(all_segments[0].id)
l = len(all_segments)
print("Min length:", len(all_segments[0].ax))
print("1st quantile length:", len(all_segments[int(l*0.25)].ax))
print("Median length:", len(all_segments[int(l*0.5)].ax))
print("3rd quantile length:", len(all_segments[int(l*0.75)].ax))
print("95 percent length:", len(all_segments[int(l*0.95)].ax))
print("Max length:", len(all_segments[-1].ax))
print("")

#Find some metrics for the detected segments.
length_segments_1axis = []
length_segments_2axis = []
length_segments_3axis = []
for segment in all_segments:
    if len(segment.axis) == 1:
        length_segments_1axis.append(len(segment.ax))
    if len(segment.axis) == 2:
        length_segments_2axis.append(len(segment.ax))
    if len(segment.axis) == 3:
        length_segments_3axis.append(len(segment.ax))

print("Number of 1-axis segments: "+str(len(length_segments_1axis)))    
print("Max 1-axis segment length: "+str(max(length_segments_1axis)))
print("Min 1-axis segment length: "+str(min(length_segments_1axis)))
print("Mean 1-axis segment length: "+str(np.mean(length_segments_1axis)))
print("Median 1-axis segment length: "+str(np.median(length_segments_1axis)))
print("")

print("Number of 2-axis segments: "+str(len(length_segments_2axis)))
print("Max 2-axis segment length: "+str(max(length_segments_2axis)))
print("Min 2-axis segment length: "+str(min(length_segments_2axis)))
print("Mean 2-axis segment length: "+str(np.mean(length_segments_2axis)))
print("Median 2-axis segment length: "+str(np.median(length_segments_2axis)))
print("")

print("Number of 3-axis segments: "+str(len(length_segments_3axis)))
print("Max 3-axis segment length: "+str(max(length_segments_3axis)))
print("Min 3-axis segment length: "+str(min(length_segments_3axis)))
print("Mean 3-axis segment length: "+str(np.mean(length_segments_3axis)))
print("Median 3-axis segment length: "+str(np.median(length_segments_3axis)))
print("")

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")