import os
import time
import numpy as np
import multiprocessing
from scipy import signal
from functools import partial

import models.data_manager
import models.segment_manager

import warnings
warnings.filterwarnings("ignore")

'''
Note: scipy.signal.correlation_lags method needs Scipy 1.6.3 to work.
'''

def compute_max_corr(segments):
    maxcorr, maxcorr_lag = np.empty((len(segments), len(segments))), np.empty((len(segments), len(segments)))
    for i in range(len(segments)):
        for j in range(len(segments)):
            a = segments[i]
            b = segments[j]
            
            normalized_a = np.float32((a - np.mean(a)) / np.std(a))
            normalized_b = np.float32((b - np.mean(b)) / np.std(b))
            
            corr = np.float32(np.correlate(normalized_a, normalized_b, 'full') / max(len(a), len(b)))
            maxcorr[i,j] = max(corr)
            
            lag = signal.correlation_lags(normalized_a.size, normalized_b.size, mode = 'full')
            maxcorr_lag[i,j] = lag[np.argmax(corr)]
            
    return maxcorr, maxcorr_lag

def compute_max_corr_parallel(segments):
    output = []
    compute_max_corr_1segment_partial = partial(compute_max_corr_1segment, segments = segments)
    
    pool = multiprocessing.Pool(processes = 16)
    o = pool.map_async(compute_max_corr_1segment_partial, segments).get()

    output.append(o)
    pool.close()
    pool.join()    

    return output

def compute_max_corr_1segment(segment, segments):
    maxcorr, maxcorr_lag = np.empty(len(segments)), np.empty(len(segments))
    for j in range(len(segments)):
        a = segment
        b = segments[j]
            
        normalized_a = np.float32((a - np.mean(a)) / np.std(a))
        normalized_b = np.float32((b - np.mean(b)) / np.std(b))
        
        corr = np.float32(np.correlate(normalized_a, normalized_b, 'full') / max(len(a), len(b)))
        maxcorr[j] = np.float32(max(corr))
        
        lag = signal.correlation_lags(normalized_a.size, normalized_b.size, mode = 'full')
        maxcorr_lag[j] = np.float16(lag[np.argmax(corr)])
        
    return maxcorr, maxcorr_lag

if __name__ == "__main__":
    start_time = time.time()

    ### Initialize data_manager and segment_manager    
    sigma = 6
    w = 100
    data_manager = data_manager.data_manager()
    
    path = "../Data/output/"
    all_data = np.load(path + "all_data.npy")
    print("Data loaded")
    
    all_segments = data_manager.load_all_segments_linux(path, sigma, w)
    for data in all_data:
        for segment in all_segments:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
                #segment.setup_gps_data(data)
                #segment.setup_timestamp(data)
    print("Segments loaded.")
    
    ### Prepare segments to compute max correlation
    segments_ax, segments_ay, segments_az = [], [], []
    for segment in all_segments:
        segments_ax.append(np.array(segment.ax))
        segments_ay.append(np.array(segment.ay))
        segments_az.append(np.array(segment.az))
    
    # Delete unnecesary variables to free memory
    del all_data
    del all_segments
    del data_manager
    del data
    
    ### Method 2 for parallel processing of max correlation:
    ### For each axis, create several parallel processes using map_async
    ### to compute different rows of the max correlation matrix (faster than method 1).
    print("Starting the computation of max correlation...")
    output_ax = compute_max_corr_parallel(segments_ax)
    np.save(os.path.join(path, 'maxcorr_ax.npy'), np.array(output_ax[:][0][:])[:,0,:])
    np.save(os.path.join(path, 'lag_ax.npy'), np.array(output_ax[:][0][:])[:,1,:])
    print("Axis x max correlation matrix computed")
    del output_ax
    del segments_ax
    
    output_ay = compute_max_corr_parallel(segments_ay)
    np.save(os.path.join(path, 'maxcorr_ay.npy'), np.array(output_ay[:][0][:])[:,0,:])
    print("Axis y max correlation matrix computed")
    del output_ay
    del segments_ay
    
    output_az = compute_max_corr_parallel(segments_az)
    np.save(os.path.join(path, 'maxcorr_az.npy'), np.array(output_az[:][0][:])[:,0,:])
    print("Axis z max correlation matrix computed")
    del output_az
    del segments_az

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")
    

