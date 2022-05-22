import os
import time
import numpy as np
import multiprocessing
from scipy import signal
from functools import partial

import models.data_manager as data_manager
import models.segment_manager as segment_manager

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
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

    i = 1
    while i < len(all_segments):
        prev = all_segments[i-1]
        curr = all_segments[i]
        if(prev.filename == curr.filename and (curr.start - prev.end) <= w/2):
            #Merge them
            new_segment = self.merge_segments(curr.filename, curr, prev)
            all_segments, _, _, _, i = self.update_segments(all_segments, curr, prev, None, new_segment, "previous")

    ### Export all segments to CSV
    data_manager.export_all_segments(all_segments, sigma, w, path)
    print("Total number of segments: "+str(len(all_segments)))


    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")