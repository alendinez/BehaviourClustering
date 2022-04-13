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
lag_ax = np.load(output_path + "lag_ax.npy")

### Save N most common behaviors
N = 10
groups = segment_manager.save_most_common_behaviors(groups_raw, N)
print(N, "most common behaviours selected")

### Find median behavior for each group
avrg_group_ax, avrg_group_ay, avrg_group_az = segment_manager.find_median_behavior(groups)

### Calc cross correlations between the medians
for i in range(N):
    for j in range(i, N):
        a_ax = avrg_group_ax[i]
        a_ay = avrg_group_ay[i]
        a_az = avrg_group_az[i]

        b_ax = avrg_group_ax[j]
        b_ay = avrg_group_ay[j]
        b_az = avrg_group_az[j]

        normalized_a_ax = np.float32((a_ax - np.mean(a_ax)) / np.std(a_ax))
        normalized_a_ay = np.float32((a_ay - np.mean(a_ay)) / np.std(a_ay))
        normalized_a_az = np.float32((a_az - np.mean(a_az)) / np.std(a_az))

        normalized_b_ax = np.float32((b_ax - np.mean(b_ax)) / np.std(b_ax))
        normalized_b_ay = np.float32((b_ay - np.mean(b_ay)) / np.std(b_ay))
        normalized_b_az = np.float32((b_az - np.mean(b_az)) / np.std(b_az))

        corr_ax = np.float32(np.correlate(normalized_a_ax, normalized_b_ax, 'full') / max(len(a_ax), len(b_ax)))
        corr_ay = np.float32(np.correlate(normalized_a_ay, normalized_b_ay, 'full') / max(len(a_ay), len(b_ay)))
        corr_az = np.float32(np.correlate(normalized_a_az, normalized_b_az, 'full') / max(len(a_az), len(b_az)))

        print(f'Maximum cross-correlations between groups {i} and {j}: {max(corr_ax)}\t{max(corr_ay)}\t{max(corr_az)}')


