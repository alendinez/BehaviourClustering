import os
import copy
import time
import random
import numpy as np
import data_manager
import segment_manager
import multiprocessing

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

if __name__ == "__main__":
    start_time = time.time()

    ### Initialize data_manager and segment_manager    
    sigma = 4
    w = 50
    mode = "mean"
    segment_manager = segment_manager.segment_manager(sigma, w, mode)
    data_manager = data_manager.data_manager()
    
    ### Load acceleration data
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
    
    all_data = []
    path = "../Data/CSV/"

    print("Starting...")
    for filename in filenames:
        # Load data and filter acceleration signals with a butterworth filter
        data = data_manager.load_data(filename, path)
        data.filter_accelerations(4, 0.4)
        all_data.append(data)
        print("Data loaded: "+filename)

    path = "../Data/output/"
    ### Load previously created acceleration segments
    all_segments = data_manager.load_all_segments(path, sigma, w)
    for data in all_data:
        for segment in all_segments:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
    print("Segments loaded.")
    
    ### Load correlation data
    maxcorr_ax = np.load(path+"maxcorr_ax.npy")
    maxcorr_ay = np.load(path+"maxcorr_ay.npy")
    maxcorr_az = np.load(path+"maxcorr_az.npy") 
    lag_ax = np.load(path+"lag_ax.npy")
    
    ### Call the group_segments function
    threshold_ax = 0.4
    threshold_ay = 0
    threshold_az = 0.1
    input_segments = copy.copy(all_segments)
    groups_raw = group_segments(input_segments, maxcorr_ax, maxcorr_ay, maxcorr_az, threshold_ax, threshold_ay, threshold_az)
    
    print("Number of raw groups: "+str(len(groups_raw)))
    path = ""
    np.save(os.path.join(path, 'groups_raw.npy'), groups_raw)
    
    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")