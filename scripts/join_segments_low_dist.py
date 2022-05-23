import sys
sys.path.insert(0,"..") ## Set path to main directory

import os
import time
import numpy as np
import multiprocessing
import csv
from scipy import signal
from functools import partial

import models.data_manager as data_manager
import models.segment_manager as segment_manager

import warnings
warnings.filterwarnings("ignore")

def export_all_segments(segments, sigma, w, path):
    fields = ['id', 'start', 'end', 'axis', 'filename']
    export_filename = path+"allsegments_sigma"+str(sigma)+"_w"+str(w)+"_joined.csv"
    
    with open(export_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for segment in segments:
            writer.writerow([segment.id, segment.start, segment.end, segment.axis, segment.filename])

if __name__ == "__main__":
    start_time = time.time()

    ### Initialize data_manager and segment_manager    
    sigma = 0.3
    w = 150
    mode = "std"
    segment_manager = segment_manager(sigma, w, mode)
    data_manager = data_manager()

    path = "../../Data/output/"

    ### Load previously created acceleration segments
    all_segments = data_manager.load_all_segments_linux(path, sigma, w)
    #all_segments.sort(key=lambda x: x.id)

    i = 0
    while i < len(all_segments) - 1:
        current_segment = all_segments[i]
        next_segment = all_segments[i+1]
        while(next_segment.filename == current_segment.filename and (next_segment.start - current_segment.end ) <= w/2):
            #Merge them
            all_segments[i] = segment_manager.merge_segments(current_segment.filename, current_segment, next_segment)
            all_segments.remove(next_segment)
            
            current_segment = all_segments[i]
            next_segment = all_segments[i+1]
                
        i = i + 1

    # Add segment id to each segment
    i = 0
    for segment in all_segments:
        segment.id = i
        i = i+1

    ### Export all segments to CSV
    export_all_segments(all_segments, sigma, w, path)
    print("Total number of segments: "+str(len(all_segments)))


    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")