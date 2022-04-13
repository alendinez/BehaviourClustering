import time
import copy
import numpy as np
from matplotlib import pyplot as plt

import models.data_manager
import models.segment_manager

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_managerager class.

### Initialize with sigma, w and mode (rest or mean).
sigma = 1
w = 100
mode = "fixed"
segment_manager = segment_manager.segment_manager(sigma, w, mode)
data_manager = data_manager.data_manager()

### Initialize a list to store the events from all the datasets.
all_segments = []

output_path = "../Data/output/"
all_data = np.load(output_path + "all_data.npy", allow_pickle = True)
print("Data loaded")

print("Starting...")
### Detect events for a given datasets
for data in all_data:
    print("Processing file:", data.filename)

    ### Find raw events for ax, ay and az.
    segments_ax = segment_manager.create_raw_segments(data.filename, data.ax, "x")
    segments_ay = segment_manager.create_raw_segments(data.filename, data.ay, "y")
    segments_az = segment_manager.create_raw_segments(data.filename, data.az, "z")
    
    ### Save initial segments into a different list to check that none of them are lost after overlapping.
    init_segments = segments_ax + segments_ay + segments_az
    print("Initial segments found: "+str(len(init_segments)))

    ### Find overlapping segments
    current_segments = copy.deepcopy(init_segments)
    current_segments = segment_manager.overlap_segments_one_direction(data.filename, current_segments, len(data.ax))
    print("Segments found after overlapping: "+str(len(current_segments)))

    ### Append the segments found in the current dataset into a list that contains the segments from ALL datasets.
    all_segments = all_segments + current_segments

# Add segment id to each segment
i = 0
for segment in all_segments:
    segment.id = i
    i = i+1

### Export all segments to CSV
data_manager.export_all_segments(all_segments, sigma, w, output_path)
print("All segments successfully exported to .csv")
print("Total number of segments: "+str(len(all_segments)))

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")


'''
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
print("")

print("Number of 2-axis segments: "+str(len(length_segments_2axis)))
print("Max 2-axis segment length: "+str(max(length_segments_2axis)))
print("Min 2-axis segment length: "+str(min(length_segments_2axis)))
print("Mean 2-axis segment length: "+str(np.mean(length_segments_2axis)))
print("")

print("Number of 3-axis segments: "+str(len(length_segments_3axis)))
print("Max 3-axis segment length: "+str(max(length_segments_3axis)))
print("Min 3-axis segment length: "+str(min(length_segments_3axis)))
print("Mean 3-axis segment length: "+str(np.mean(length_segments_3axis)))
'''