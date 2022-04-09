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
sigma = 6
w = 100
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)
data_manager = data_manager.data_manager()

### Initialize a list to store the events from all the datasets.
all_data = []
all_segments = []

### Define the names of the datasets that we will use
filenames = ['7501394_rec16112018_PRincon_PHAAET_S1_',
            '7501709_rec18112018_PRincon_PHAAET_S1_',
            '7501755_rec27112018_PRincon_PHAAET_AXY_S1', 
            '8200163_rec14052019_PRoque_PHAAET_S1',
            '8200445_rec29042019_PRincon_PHAAET_S1',
            '8200473_rec24052019_PRincon_PHAAET_S1',
            '8200487_rec04052019_PRincon_PHAAET_S1',
            '8200718_PHAAET_rec08032019_PRincon',
            '8201653_PHAAET_rec21012021_ICima_Ninho39_36_S1',
            '8201667_PHAAET_rec21012021_ICima_Ninho68_21_S1',
            '8201720_rec31122020_ICima_PHAAET_ninho 71_21_S11_S1',
            '8201959_rec29122020_ICima_PHAAET_ninho 31_36_S1']

print("Starting...")
### Detect events for a given datasets
for filename in filenames:
    
    path = "../Data/CSV/"
    
    # Load data and filter acceleration signals with a butterworth filter
    initial_data = data_manager.load_data(filename, path)
    current_data = copy.deepcopy(initial_data)
    current_data.filter_accelerations(4, 0.4)
    all_data.append(current_data)
    print("Data loaded: "+filename)

    ### Find raw events for ax, ay and az.
    segments_ax = segment_manager.create_raw_segments(filename, current_data.ax, "x", "mean")
    segments_ay = segment_manager.create_raw_segments(filename, current_data.ay, "y", "mean")
    segments_az = segment_manager.create_raw_segments(filename, current_data.az, "z", "mean")
    
    ### Save initial segments into a different list to check that none of them are lost after overlapping.
    init_segments = segments_ax + segments_ay + segments_az
    print("Initial segments found: "+str(len(init_segments)))

    ### Find overlapping segments
    current_segments = copy.deepcopy(init_segments)
    current_segments = segment_manager.overlap_segments_one_direction(filename, current_segments, len(current_data.ax))
    print("Segments found after overlapping: "+str(len(current_segments)))

    ### Append the segments found in the current dataset into a list that contains the segments from ALL datasets.
    all_segments = all_segments + current_segments

# Add segment id to each segment
i = 0
for segment in all_segments:
    segment.id = i
    i = i+1

### Export all segments to CSV
export_path = "../Data/output"

data_manager.export_all_segments(all_segments, sigma, w, export_path)
print("All segments successfully exported to .csv.")
print("")
   
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