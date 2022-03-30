import os
import time
import numpy as np
import data_manager
import segment_manager
import multiprocessing
from scipy import signal
from functools import partial

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
    
    pool = multiprocessing.Pool(processes = 14)
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
    sigma = 8
    w = 50
    mode = "mean"
    segment_manager = segment_manager.segment_manager(sigma, w, mode)
    data_manager = data_manager.data_manager()
    
    ### Load acceleration data
    filenames = ['8201720_PHAAET_rec31122020_ICima_ninho 71_21_S1']
    
    all_data = []
    for filename in filenames:
        datapath =''+filename+'\\'   
        # Load data and filter acceleration signals with a butterworth filter
        #data = data_manager.load(filename, datapath)
        data = data_manager.load_data_gps(filename, datapath)
        data.filter_accelerations(4, 0.4)
        all_data.append(data)
        print("Data loaded: "+filename)
    
    ### Load previously created acceleration segments
    path = ""

    

    #all_segments = data_manager.load_all_segments(path, sigma, w)
    all_segments = np.load(path+"allsegments_gps.npy", allow_pickle = True)
    for data in all_data:
        for segment in all_segments:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
                #segment.setup_gps_data(data)
                #segment.setup_timestamp(data)
    print("Segments loaded.")
    
    ### Prepare segments to compute max correlation
    i = 0
    segments_ax, segments_ay, segments_az = [], [], []
    for segment in all_segments:
        segment.id = i
        segments_ax.append(np.array(segment.ax))
        segments_ay.append(np.array(segment.ay))
        segments_az.append(np.array(segment.az))
        i = i + 1
    segments_a = [segments_ax, segments_ay, segments_az]   
    
    '''
    ### Method 1 for parallel processing of max correlation:
    ### Create 3 parallel process, one for each acceleration axis.
    
    ### Send segments to the method that computes the max correlation using parallel processing.
    pool = multiprocessing.Pool(processes = 3)
    output_ax, output_ay, output_az = pool.map(compute_max_corr, segments_a)
    pool.close()
    pool.join()
    
    ### Divide each output into correlation and lag.
    maxcorr_ax, lag_ax = output_ax[0], output_ax[1]
    maxcorr_ay, lag_ay = output_ay[0], output_ay[1]
    maxcorr_az, lag_az = output_az[0], output_az[1]
    ### Save correlation and lag into numpy format
    np.save(os.path.join(path, 'maxcorr_ax.npy'), maxcorr_ax)
    np.save(os.path.join(path, 'maxcorr_ay.npy'), maxcorr_ay)
    np.save(os.path.join(path, 'maxcorr_az.npy'), maxcorr_az)
    np.save(os.path.join(path, 'lag_ax.npy'), lag_ax)
    '''
    
    ### Method 2 for parallel processing of max correlation:
    ### For each axis, create several parallel processes using map_async
    ### to compute different rows of the max correlation matrix (faster than method 1).
    output_ax = compute_max_corr_parallel(segments_ax)
    np.save(os.path.join(path, 'output_ax.npy'), output_ax)
    output_ay = compute_max_corr_parallel(segments_ay)
    np.save(os.path.join(path, 'output_ay.npy'), output_ay)
    output_az = compute_max_corr_parallel(segments_az)
    np.save(os.path.join(path, 'output_az.npy'), output_az)
    
    ### Divide the output into max correlation and lag
    a = np.array(output_ax[:][0][:])[:,0,:]
    np.save(os.path.join(path, 'maxcorr_ax.npy'), a)
    
    a = np.array(output_ay[:][0][:])[:,0,:]
    np.save(os.path.join(path, 'maxcorr_ay.npy'), a)
    
    a = np.array(output_az[:][0][:])[:,0,:]
    np.save(os.path.join(path, 'maxcorr_az.npy'), a)
    
    a = np.array(output_ax[:][0][:])[:,1,:]
    np.save(os.path.join(path, 'lag_ax.npy'), a)

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")
    

