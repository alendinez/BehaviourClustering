import copy
import time
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

import models.segment_manager as segment_manager
import models.data_manager as data_manager
import models.segment as segment

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_manager and segment_manager    
sigma = 6
w = 100
mode = "mean"
segment_manager = segment_manager(sigma, w, mode)
data_manager = data_manager()

output_path = "../Data/output/"
all_data = np.load(output_path + "all_data.npy", allow_pickle = True)
print("Data loaded")

groups_raw = np.load(output_path + "groups_raw.npy", allow_pickle = True)

### Save N most common behaviors
N = 10
groups = segment_manager.save_most_common_behaviors(groups_raw, N)
print(N, "most common behaviours selected")

### Load correlation data
maxcorr_ax = np.load(output_path + "maxcorr_ax.npy")
maxcorr_ay = np.load(output_path + "maxcorr_ay.npy")
maxcorr_az = np.load(output_path + "maxcorr_az.npy") 

### Plot the correlation of the maximum exponent of group 3 with groups 3, 4 and 7
min_len_group = min(len(groups[3]), len(groups[4]), len(groups[7]))

main_id = groups[3][0].id

segments_group_3 = [segment.id for segment in groups[3][1:min_len_group+1]]
segments_group_4 = [segment.id for segment in groups[4][:min_len_group]]
segments_group_7 = [segment.id for segment in groups[7][:min_len_group]]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in segments_group_3:
    xs = maxcorr_ax[main_id, i]
    ys = maxcorr_ay[main_id, i]
    zs = maxcorr_az[main_id, i]
    ax.scatter(xs, ys, zs, marker='o', c='red')

for i in segments_group_4:
    xs = maxcorr_ax[main_id, i]
    ys = maxcorr_ay[main_id, i]
    zs = maxcorr_az[main_id, i]
    ax.scatter(xs, ys, zs, marker='o', c='green')

for i in segments_group_7:
    xs = maxcorr_ax[main_id, i]
    ys = maxcorr_ay[main_id, i]
    zs = maxcorr_az[main_id, i]
    ax.scatter(xs, ys, zs, marker='o', c='blue')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

