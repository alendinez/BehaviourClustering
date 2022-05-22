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

'''
Note: scipy.signal.correlation_lags method needs Scipy 1.6.3 to work.
'''
def normalized_cc(s1, s2, norm1=-1., norm2=-1.):

    sz = s1.shape[0]
    d = s1.shape[1]
    # Compute fft size based on tip from https://stackoverflow.com/questions/14267555/
    fft_sz = 1 << (2 * sz - 1).bit_length()

    denom = 0.

    if norm1 < 0.:
        norm1 = np.linalg.norm(s1)
    if norm2 < 0.:
        norm2 = np.linalg.norm(s2)
    
    denom = norm1 * norm2
    if denom < 1e-9:  # To avoid NaNs
        denom = np.inf

    cc = np.real(np.fft.ifft(np.fft.fft(s1, fft_sz, axis=0) *
                                   np.conj(np.fft.fft(s2, fft_sz, axis=0)), axis=0))
    cc = np.vstack((cc[-(sz-1):], cc[:sz]))
    return np.real(cc).sum(axis=-1) / denom

def compute_max_corr_parallel(segments, norms):
    output = []
    compute_max_corr_1segment_partial = partial(compute_max_corr_1segment, segments = segments, norms = norms)
    
    pool = multiprocessing.Pool(processes = 15)
    o = pool.map_async(compute_max_corr_1segment_partial, range(len(segments))).get()

    output.append(o)
    pool.close()
    pool.join()    

    return output

def compute_max_corr_1segment(i, segments, norms):
    maxcorr = np.empty(len(segments))
    for j in range(len(segments)):
        if j <= i: maxcorr[j] = 0.
        else: 
            a = segments[i]
            b = segments[j]

            if(len(a) < len(b)):
                a,b = b,a
            
            cc = np.float32(normalized_cc(a,b,norms[i],norms[j]))
            maxcorr[j] = cc.max()
        
    return maxcorr

if __name__ == "__main__":
    start_time = time.time()

    ### Initialize data_manager and segment_manager    
    sigma = 0.3
    w = 100
    mode = "std"
    segment_manager = segment_manager(sigma, w, mode)
    data_manager = data_manager()

    path = "../Data/output/"
    all_data = np.load(path + "all_data.npy", allow_pickle = True)
    print("Data loaded")

    ### Load previously created acceleration segments
    all_segments = data_manager.load_all_segments_linux(path, sigma, w)
    for data in all_data:
        for segment in all_segments:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
    print("Acceleration data set")
    '''
    ### Segments filtering
    segments_out = [idx for idx in range(len(all_segments)) if len(all_segments[idx].ax) > 5000]
    segments_excluded = 1 - ((len(all_segments) - len(segments_out)) / len(all_segments))
    segments_out.sort(reverse=True)
    for idx in segments_out:
        all_segments.pop(idx)
    print("Segments filtered:", segments_excluded * 100, "%")
    
    X = segment_manager.format_segments(all_segments)
    print("Segments dataset shape:", X.shape)

    # Normalize the data
    mean_t = np.nanmean(X, axis=1, keepdims=True)
    std_t = np.nanstd(X, axis=1, keepdims=True)
    std_t[std_t == 0.] = 1.

    X = (X - mean_t) / std_t
    print("Data normalized")

    # Change NaNs with 0. For the computation of the algorithm is indifferent.
    X[np.isnan(X)] = 0.
    '''
    X = segment_manager.format_segments(all_segments)
    norms = []
    for i in range(len(X)):
        X[i] = (X[i] - np.mean(X[i], axis=0, keepdims=True)) / np.std(X[i], axis=0, keepdims=True)
        norms.append(np.linalg.norm(X[i]))
    
    #norms = np.linalg.norm(X, axis=(1, 2))
    norms = np.array(norms)

    # Delete unnecesary variables to free memory
    del all_data
    del all_segments
    del data_manager
    del data
    #del segments_out
    
    ### Method 2 for parallel processing of max correlation:
    ### For each axis, create several parallel processes using map_async
    ### to compute different rows of the max correlation matrix (faster than method 1).
    print("Starting the computation of max correlation...")
    maxcorr = np.array(compute_max_corr_parallel(X, norms))[0]
    maxcorr = maxcorr + maxcorr.T # Copy the upper triangle in the lower triangle
    np.fill_diagonal(maxcorr, 1.)
    np.save(os.path.join(path, 'maxcorr_03_100.npy'), maxcorr)

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")
    

