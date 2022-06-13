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

import models.data as dt
import models.data_manager as data_manager
import models.segment_manager as segment_manager
import models.segment as sgmnt
import models.KShapeVariableLength as KShapeVariableLength

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()


output_path = "../../Data/output/"

groups = np.load(output_path + "groups_raw_16_3.npy", allow_pickle = True)
pipeline_output = np.load(output_path + 'pipeline_output.npy', allow_pickle = True)

initial_groups_lengths = [len(group) for group in groups]
print("Initial groups lengths:", initial_groups_lengths)

segments_train = pipeline_output[0]
segments_test = pipeline_output[1]
labels_train = pipeline_output[2]
labels_test = pipeline_output[3]
prediction_train = pipeline_output[4]
prediction_test = pipeline_output[5]

assert len(segments_train) == len(labels_train) and len(segments_train) == len(prediction_train)
assert len(segments_test) == len(labels_test) and len(segments_test) == len(prediction_test)

augmented_segments_missclassified = 0
segments_missclassified = 0

### Train segments
for i in range(len(segments_train)):
    if labels_train[i] != prediction_train[i]:
        current_segment = segments_train[i]
        if current_segment.id is None:
            augmented_segments_missclassified += 1
        else:
            for sgmnt in groups[labels_train[i]]:
                if sgmnt == current_segment.id:
                    groups[labels_train[i]].remove(sgmnt)
                    break
            groups[prediction_train[i]].append(current_segment.id)
        segments_missclassified += 1

### Test segments
for i in range(len(segments_test)):
    if labels_test[i] != prediction_test[i]:
        current_segment = segments_test[i]
        if current_segment.id is None:
            augmented_segments_missclassified += 1
        else:
            for sgmnt in groups[labels_test[i]]:
                if sgmnt == current_segment.id:
                    groups[labels_test[i]].remove(sgmnt)
                    break
            groups[prediction_test[i]].append(current_segment.id)
        segments_missclassified += 1

final_groups_lengths = [len(group) for group in groups]
print("Final groups lengths:", final_groups_lengths)
print("Segments missclassified:", segments_missclassified)
print("Augmented segments missclassified:", augmented_segments_missclassified)

del segments_train
del segments_test
del labels_train
del labels_test
del prediction_train
del prediction_test
del pipeline_output

'''
groups_idxs = []
for group in groups:
    temp = []
    for sgmnt in group:
        temp.append(sgmnt)
    groups_idxs.append(temp)
'''
np.save(os.path.join(output_path, 'groups_raw_16_4.npy'), groups)


finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")