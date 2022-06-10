import sys
sys.path.insert(0,"..") ## Set path to main directory

import copy
import time
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
import multiprocessing
from scipy import signal
from functools import partial

import models.segment_manager as segment_manager
import models.data_manager as data_manager
import models.segment as segment
import models.KShapeVariableLength as KShapeVariableLength

import warnings
warnings.filterwarnings("ignore")

def compute_max_corr_1segment(segment, segments):
    maxcorr, maxcorr_lag = np.empty(len(segments)), np.empty(len(segments))
    for j in range(len(segments)):
        a = segment
        b = segments[j]
            
        normalized_a = np.float32((a - np.mean(a)) / np.std(a))
        normalized_b = np.float32((b - np.mean(b)) / np.std(b))
        
        corr = np.float32(signal.correlate2d(a, b, mode = "full")[:,2] / (np.linalg.norm(a)*np.linalg.norm(b)))
        maxcorr[j] = np.float32(max(corr))
        
        lag = signal.correlation_lags(normalized_a.size, normalized_b.size, mode = 'full')
        maxcorr_lag[j] = np.float16(lag[np.argmax(corr)])
        
    return maxcorr, maxcorr_lag

def centroid_selection(cross_dists, labels, k):
    # Select submatrix of cross dists with label k
    idxs = (labels == k).nonzero()[0]
    dists = 1. - cross_dists[np.ix_(idxs,idxs)]
    # Get the one with max value
    centroid = idxs[dists.sum(axis=1).argmin()]
    return centroid

start_time = time.time()

### Initialize data_manager and segment_manager    
sigma = 0.3
w = 150
mode = "std"
segment_manager = segment_manager(sigma, w, mode)
data_manager = data_manager()

output_path = "../../Data/output/"

### Load all data
all_data = np.load(output_path + "all_data.npy", allow_pickle = True)
print("Data loaded")

### Load previously created acceleration segments
all_segments = data_manager.load_all_segments_linux(output_path, sigma, w)
for data in all_data:
    for segment in all_segments:
        if segment.filename == data.filename:
            segment.setup_acceleration(data)
print("Acceleration data set")

### Segments filtering
segments_copy = all_segments.copy()
segments_copy.sort(key=lambda x: len(x.ax), reverse=True)
l = len(segments_copy)
segments_in = [x.id for x in segments_copy[int(l*0.05):]]
segments_in.sort()
assert segments_in[0] == min(segments_in)
print("Segments filtered")

maxcorr = np.load(output_path + f"maxcorr_{sigma}_{w}.npy")
maxcorr = maxcorr[np.ix_(segments_in, segments_in)]
print("Max correlation matrix loaded:", maxcorr.shape)

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

groups = np.load(output_path + "groups_raw_12_4.npy", allow_pickle = True)

### Get labels array
labels = np.empty(len(maxcorr), dtype=int)
for i in range(len(groups)):
    for idx in groups[i]:
        labels[idx] = i

### Get centroids
centroids = []
n_clusters = len(groups)
avrg_group_ax, avrg_group_ay, avrg_group_az = [], [], []
cluster_centers = np.empty(n_clusters, dtype=int)
for i in range(n_clusters):
    cluster_centers[i] = centroid_selection(maxcorr, labels, i)

data = np.empty((len(cluster_centers), len(cluster_centers)))
for i in range(len(cluster_centers)):
    for j in range(len(cluster_centers)):
        data[i,j] = maxcorr[cluster_centers[i]][cluster_centers[j]]

fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(data)
ax.set_xticks(np.arange(len(cluster_centers)), minor=False)
ax.set_yticks(np.arange(len(cluster_centers)), minor=False)

for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

plt.title("Group correlations")
plt.show()


