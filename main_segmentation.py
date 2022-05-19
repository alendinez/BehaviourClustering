import time
import copy
import numpy as np
from matplotlib import pyplot as plt

import models.data_manager as data_manager
import models.segment_manager as sm

import warnings
warnings.filterwarnings("ignore")

params = [("std", 100, 0.3),("std", 150, 0.3),("std", 200, 0.3)]

output_path = "../Data/output/"
all_data = np.load(output_path + "all_data.npy", allow_pickle = True)
print("Data loaded")

data_manager = data_manager()

for mode, w, sigma in params:
    start_time = time.time()

    ### Initialize with sigma, w and mode (rest or mean).
    #sigma = 1
    #w = 100
    #mode = "fixed"
    segment_manager = sm(sigma, w, mode)

    ### Initialize a list to store the events from all the datasets.
    all_segments = []

    print(f"Starting segmentation with w={w}, sigma={sigma}, mode='{mode}'...")
    ### Detect events for a given datasets
    for data in all_data:

        ### Find raw events for ax, ay and az.
        segments_ax = segment_manager.create_raw_segments(data.filename, data.ax, "x")
        segments_ay = segment_manager.create_raw_segments(data.filename, data.ay, "y")
        segments_az = segment_manager.create_raw_segments(data.filename, data.az, "z")
        
        ### Save initial segments into a different list to check that none of them are lost after overlapping.
        init_segments = segments_ax + segments_ay + segments_az

        ### Find overlapping segments
        current_segments = copy.deepcopy(init_segments)
        current_segments = segment_manager.overlap_segments_one_direction(data.filename, current_segments, len(data.ax))

        ### Append the segments found in the current dataset into a list that contains the segments from ALL datasets.
        all_segments = all_segments + current_segments

    # Add segment id to each segment
    i = 0
    for segment in all_segments:
        segment.id = i
        i = i+1

    ### Export all segments to CSV
    data_manager.export_all_segments(all_segments, sigma, w, output_path)
    print("Total number of segments: "+str(len(all_segments)))


    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")