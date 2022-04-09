import os
import copy
import time
import random
import numpy as np
import multiprocessing

import models.data_manager
import models.segment_manager

import warnings
warnings.filterwarnings("ignore")

def group_segments(input_segments, corr_ax, corr_ay, corr_az, threshold_ax, threshold_ay, threshold_az):
### Add a global index to each segment from 0 to len(segments)
    segments = copy.copy(input_segments)
    
    ### Take one segment e1, if the next one has a correlation higher than threshold, we put them into a separate list. 
    ### Repeat until there are no more segments with correlation higher than threshold for e1.
    
    similar_segments = []       
    i = 0
    while i < len(segments):
        current_segment = segments[i]
        temp_similar_segments = [current_segment]
        segments.remove(segments[i])
        
        j = i+1
        while j < len(segments):
            next_segment = segments[j]
            c_ax = corr_ax[current_segment.id, next_segment.id]
            c_ay = corr_ay[current_segment.id, next_segment.id]
            c_az = corr_az[current_segment.id, next_segment.id]
            
            if float(c_ax) >= threshold_ax and float(c_ay) >= threshold_ay and float(c_az) >= threshold_az:
                temp_similar_segments.append(next_segment)
                segments.remove(segments[j])
                j = i+1
            else:
                j = j+1
                    
        else:
            similar_segments.append(temp_similar_segments)
            i = i+1
            
    return similar_segments 

def group_segments_max_first(input_segments, corr_ax, corr_ay, corr_az, threshold_ax, threshold_ay, threshold_az):
    segments = copy.copy(input_segments)

    ### Set diagonals of correlation matrices to 0
    np.fill_diagonal(corr_ax, 0.0)
    np.fill_diagonal(corr_ay, 0.0)
    np.fill_diagonal(corr_az, 0.0)

    ### Stack the correlation matrices
    corr = np.dstack((corr_ax, corr_ay, corr_az))
    
    similar_segments = []    
    max_corr_val = 1
    ### len(segments) - 1 because the last segment is completely erased before it's turn
    while segments.count(None) < len(segments) - 1 and max_corr_val >= max(threshold_ax, threshold_ay, threshold_az):
        print(len(segments) - segments.count(None), 'segments left to be grouped...')
        ### Get index of the segment with more correlation
        index_1, index_2, index_3 = np.unravel_index(corr.argmax(), corr.shape)
        max_corr_val = corr[index_1, index_2, index_3]

        temp_similar_segments = [segments[index_1], segments[index_2]]
        segments[index_1], segments[index_2] = None, None

        ### Get the indices of the segments above the threshold in the 3 axis
        c_ax_1 = np.where(corr[index_1,:,0] >= threshold_ax)
        c_ay_1 = np.where(corr[index_1,:,1] >= threshold_ay)
        c_az_1 = np.where(corr[index_1,:,2] >= threshold_az)

        c_ax_2 = np.where(corr[index_2,:,0] >= threshold_ax)
        c_ay_2 = np.where(corr[index_2,:,1] >= threshold_ay)
        c_az_2 = np.where(corr[index_2,:,2] >= threshold_az)

        corr_indices_1 = np.intersect1d(c_ax_1, c_ay_1, c_az_1)
        corr_indices_2 = np.intersect1d(c_ax_2, c_ay_2, c_az_2)
        corr_indices = np.intersect1d(corr_indices_1, corr_indices_2)

        ### Set to 0 in the correlation matrix the current segment
        corr[index_1], corr[:,index_1], corr[index_2], corr[:,index_2] = 0.0, 0.0, 0.0, 0.0

        ### Add the segments to the group
        for i in corr_indices:
            temp_similar_segments.append(segments[i])
            segments[i] = None
            ### Set to 0 in the correlation matrix the segments already grouped
            corr[i] = 0.0
            corr[:,i] = 0.0

        ### Append group
        similar_segments.append(temp_similar_segments)
            
    return similar_segments 

if __name__ == "__main__":
    start_time = time.time()

    ### Initialize data_manager and segment_manager    
    sigma = 6
    w = 100
    mode = "mean"
    segment_manager = segment_manager.segment_manager(sigma, w, mode)
    data_manager = data_manager.data_manager()
    
    path = "../Data/output/"
    all_data = np.load(path + "all_data.npy")
    print("Data loaded")

    ### Load previously created acceleration segments
    all_segments = data_manager.load_all_segments_linux(path, sigma, w)
    for data in all_data:
        for segment in all_segments:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
    
    ### Load correlation data
    maxcorr_ax = np.load(path + "maxcorr_ax.npy")
    maxcorr_ay = np.load(path + "maxcorr_ay.npy")
    maxcorr_az = np.load(path + "maxcorr_az.npy") 
    
    ### Call the group_segments function
    threshold_ax = 0.3
    threshold_ay = 0.3
    threshold_az = 0.3
    input_segments = copy.copy(all_segments)
    groups_raw = group_segments_max_first(input_segments, maxcorr_ax, maxcorr_ay, maxcorr_az, threshold_ax, threshold_ay, threshold_az)
    
    # Delete unnecesary variables to free memory
    del maxcorr_ax
    del maxcorr_ay
    del maxcorr_az
    del all_segments
    del input_segments
    del all_data

    print("Number of raw groups: "+str(len(groups_raw)))
    groups_raw = np.array(groups_raw, dtype="object")
    np.save(os.path.join(path, 'groups_raw.npy'), groups_raw)
    
    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")