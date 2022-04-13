import copy
import numpy as np
from scipy import signal
import more_itertools as mit

import models.segment as sgmnt

class segment_manager():
    def __init__(self, sigma, w, mode = "main"):
        
        self.sigma = sigma # Beyond sigma*stdev we consider that we have an segment or a potential behavior.
        self.w = w # Size of the window that we will use to find overlapping segments.
        self.mode = mode
        
    '''

    The function create_raw_segments creates raw segments from an acceleration signal. We define these initial segments as parts of the signal where:
        
        a < mean(a) - sigma*stdev(a) or a > mean(a) + sigma*stdev(a)
        
    Where a is the acceleration signal and sigma is the parameter that defines how much the acceleration has to deviate from its mean to
    segment it by that point.
    
    These segments are of the following form:
        
        segment = [start, end, axis]
    
    Where start is the sample at which the segment starts, end is the sample at which the segment ends, 
    and axis is the axis where the segment was initially created,
    
    '''    
    def create_raw_segments(self, filename, a, axis):
        stdev = np.std(a)
    
        if self.mode == "mean":
            mean = np.mean(a)
            index_segments = np.where((a >= mean + self.sigma*stdev) | (a <= mean - self.sigma*stdev))
        if self.mode == "rest":
            index_segments = np.where((a >= self.sigma*stdev) | (a <= -self.sigma*stdev))
        if self.mode == "fixed":
            index_segments = np.where((a >= self.sigma) | (a <= -self.sigma))
        
        index_segments = np.array(index_segments)[0]
        raw_segments = np.array([tuple(group) for group in mit.consecutive_groups(sorted(index_segments))])

        segments = []
        for segment in raw_segments:
            segments.append(sgmnt(segment[0], segment[len(segment)-1], axis, filename))
            
        return segments
    
    '''
    
    The function overlap_segments attempts to convert the raw segments into segments that correspond to behaviors in a certain temporal scale.
    
    Given a list of segments sorted by their starting sample and given a current segment inside that list:
        - This function checks if applying a window of size w there is an overlapping with the previous and/or next segments.
        - If that's the case, it adds the window to the current segment and merges it with the previous or next segment (the one that overlaps) and repeat the previous step.
        - If
        - If not, go to the next segment and repreat the first step.
            
            For example, given a segment s1 = [start1, end1, "x"] and its previous segment s2 = [start2, end2, "yz"] and a window of size w.
                
                The function checks if applying the window to s1 there is an overlapping with s2:
                    start1' = start1 - w/2, end1' = end1 + w/2
                    s1' = [start1', end1', "x"]
                    
                If there is an overlapping, the function mergers both segments and repeats the process. Therefore:
                    s3 = [min(start1', start2, max(end1', end2), "xyz")]
   '''             
                    
    def overlap_segments(self, filename, segments, signal_length):
        segments.sort(key=lambda segment: segment.start)
        i = 0
        while i < len(segments):
    
            current_segment = segments[i]
            
            ### Find previous and next segments
            previous_segment = self.find_prev_segment(current_segment, segments)
            next_segment = self.find_next_segment(current_segment, segments)
            
            ### Check if there is overlappings between current segment and previous and/or next segments
            previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
            next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
            
            ### If there is no overlappings, check if applying the window there are overlappings
            if previous_overlapping != True and next_overlapping != True:
                current_segment == self.apply_window(current_segment, signal_length)
            
                previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                
            ### While previous or next segment overlap with current segment:
            while next_overlapping == True or previous_overlapping == True:
                while previous_overlapping == True:
                    new_segment = self.merge_segments(filename, current_segment, previous_segment)
                    
                    ### Check if the segment indexes have changed. If so, add window length
                    #segment_changed = self.HasSegmentChanged(current_segment, new_segment)
                    
                    segments, current_segment, next_segment, previous_segment, i = self.update_segments(segments, current_segment, previous_segment, next_segment, new_segment, "previous")
                    previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                    next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                    
                    if previous_overlapping != True and next_overlapping != True:
                        current_segment = self.apply_window(current_segment, signal_length)
                        previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                        next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                        
                while next_overlapping == True:
                    new_segment = self.merge_segments(filename, current_segment, next_segment)
                    
                    ### Check if the segment start or end have changed. If so, check if applying the window there will be an overlap with previous or next segments. 
                    #segment_changed = self.HasSegmentChanged(current_segment, new_segment)
                    
                    segments, current_segment, next_segment, previous_segment, i = self.update_segments(segments, current_segment, previous_segment, next_segment, new_segment, "next")
                    previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                    next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                    
                    if previous_overlapping != True and next_overlapping != True:
                        current_segment = self.apply_window(current_segment, signal_length)
                        previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                        next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                        
                               
            
            else:
                i = i + 1
                
        return segments

    def overlap_segments_one_direction(self, filename, segments, signal_length):
        segments.sort(key=lambda segment: segment.start)
        i = 0
        while i < len(segments):
    
            current_segment = segments[i]
            
            ### Find previous and next segments
            previous_segment = self.find_prev_segment(current_segment, segments)
            next_segment = self.find_next_segment(current_segment, segments)
            
            ### Check if there is overlappings between current segment and previous and/or next segments
            previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
            next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
            
            ### If there is no overlappings, check if applying the window there are overlappings
            if previous_overlapping != True and next_overlapping != True:
                current_segment == self.apply_window(current_segment, signal_length)
            
                previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                
            ### No need to iter backwards because we already applied the window to the previous segment
            if previous_overlapping == True:
                new_segment = self.merge_segments(filename, current_segment, previous_segment)
                
                ### Check if the segment indexes have changed.
                segments, current_segment, next_segment, previous_segment, i = self.update_segments(segments, current_segment, previous_segment, next_segment, new_segment, "previous")
                previous_overlapping = False
                    
            while next_overlapping == True:
                new_segment = self.merge_segments(filename, current_segment, next_segment)
                
                ### Check if the segment start or end have changed. If so, check if applying the window there will be an overlap with previous or next segments. 
                #segment_changed = self.HasSegmentChanged(current_segment, new_segment)
                
                segments, current_segment, next_segment, previous_segment, i = self.update_segments(segments, current_segment, previous_segment, next_segment, new_segment, "next")
                next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                
                if next_overlapping != True:
                    current_segment = self.apply_window_right(current_segment, signal_length)
                    next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                        
                               
            
            else:
                i = i + 1
                
        return segments
   
    
    ### Apply a window of size w to a given segment in a way that an segment (start, end) becomes (start - w/2, end + w/2).
    def apply_window(self, segment, signal_length):
        if segment.start - self.w/2 < 0:
            segment.start = 0
        else:
            segment.start = segment.start - self.w/2
            
        if segment.end + self.w/2 > signal_length:
            segment.end = signal_length
        else:
            segment.end = segment.end + self.w/2
            
        return segment  

    ### Apply a window of size w/2 to a given segment in a way that an segment (start, end) becomes (start - w/2, end).
    def apply_window_left(self, segment, signal_length):
        if segment.start - self.w/2 < 0:
            segment.start = 0
        else:
            segment.start = segment.start - self.w/2
            
        return segment  

    ### Apply a window of size w/2 to a given segment in a way that an segment (start, end) becomes (start, end + w/2).
    def apply_window_right(self, segment, signal_length):            
        if segment.end + self.w/2 > signal_length:
            segment.end = signal_length
        else:
            segment.end = segment.end + self.w/2
            
        return segment   
    
    ### Method to merge two overlapping segments from the same dataset 
    def merge_segments(self, filename, current_segment, overlapping_segment): 
        start = min([current_segment.start, overlapping_segment.start])
        end = max([current_segment.end, overlapping_segment.end])
        axis = ''.join(sorted(set(current_segment.axis + overlapping_segment.axis)))
        new_segment = sgmnt(start, end, axis, filename)
        
        return new_segment
    
    
    ### Method to find the previous segment given a current segment and a list of segments that contains it.
    def find_prev_segment(self, current_segment, segments):
        if segments.index(current_segment)-1 >= 0:
            previous_segment = segments[segments.index(current_segment)-1]
        else:
            previous_segment = None
        return previous_segment
    
    
    ### Method to find the next segment given a current segment and a list of segments that contains it.
    def find_next_segment(self, current_segment, segments):
        if segments.index(current_segment)+1 <= len(segments)-1:
            next_segment = segments[segments.index(current_segment)+1]
        else:
            next_segment = None
        return next_segment
    
    
    ### Check if two segments are overlapping.
    def are_segments_overlapping(self, current_segment, other_segment):
        if other_segment != None and current_segment != None:
            starts = [current_segment.start, other_segment.start]
            ends = [current_segment.end, other_segment.end]
    
            is_overlapping = True        
            for start in starts:
                for end in ends:
                    if start > end:
                        is_overlapping = False
        else:
            is_overlapping = False
                
        return is_overlapping
    
    
    ### Check if the start or the end of an segment have changed.
    def has_segment_changed(self, current_segment, new_segment):
        segment_changed = False
        if (current_segment.start != new_segment.start) or (current_segment.end != new_segment.end):
            segment_changed = True
        
        return segment_changed
    
    
    ### Update segments list
    def update_segments(self, segments, current_segment, previous_segment, next_segment, new_segment, previous_or_next):
        try:
            segments[segments.index(current_segment)] = new_segment
        except:
            print("Error while reassigning the new segment to its new position.")
            print(new_segment)
        
        if previous_or_next == "previous":
            try:
                segments.remove(previous_segment)
            except:
                print("Error while trying to remove previous segment.")
                print(previous_segment)
                
        elif previous_or_next == "next":
            try:
                segments.remove(next_segment)
            except:
                print("Error while trying to remove next segment.")
                print(next_segment)
                
        segments.sort(key=lambda segment: segment.start)
        current_segment = new_segment
        previous_segment = self.find_prev_segment(current_segment, segments)
        next_segment = self.find_next_segment(current_segment, segments)
        i = segments.index(current_segment)
        return segments, current_segment, next_segment, previous_segment, i

    
    ### Removes all the segments of size smaller than threshold.
    def remove_short_segments(self, segments, min_segment_size):
        new_segments = []
        for segment in segments:
            length = segment.end - segment.start
            if length > min_segment_size:
                new_segments.append(segment)
                
        return new_segments
    
    
    ### Test to check that every segment has the right axis label assigned.
    def test_tag_coherence(self, segments, data):
        
        if self.mode == "mean":
            mean_ax = np.mean(data.ax)
            stdev_ax = np.std(data.ax)
            plus_ax = mean_ax + self.sigma*stdev_ax
            minus_ax = mean_ax - self.sigma*stdev_ax
            
            mean_ay = np.mean(data.ay)
            stdev_ay = np.std(data.ay)
            plus_ay = mean_ay + self.sigma*stdev_ay
            minus_ay = mean_ay - self.sigma*stdev_ay
            
            mean_az = np.mean(data.az)
            stdev_az = np.std(data.az)
            plus_az = mean_az + self.sigma*stdev_az
            minus_az = mean_az - self.sigma*stdev_az
        
        if self.mode == "rest":
            stdev_ax = np.std(data.ax)
            plus_ax = 0 + self.sigma*stdev_ax
            minus_ax = 0 - self.sigma*stdev_ax
            
            stdev_ay = np.std(data.ay)
            plus_ay = 0 + self.sigma*stdev_ay
            minus_ay = 0 - self.sigma*stdev_ay
            
            stdev_az = np.std(data.az)
            plus_az = 0 + self.sigma*stdev_az
            minus_az = 0 - self.sigma*stdev_az
        
        number_of_errors = 0
        for segment in segments:
            error_found = 0
            
            if (any(point > plus_ax or point < minus_ax for point in segment.ax)) and ("x" not in segment.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in segment.ax:
                        if (point > plus_ax or point < minus_ax):
                            print("Coherence error.")
                            break
                    
            if (any(point > plus_ay or point < minus_ay for point in segment.ay)) and ("y" not in segment.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in segment.ay:
                        if (point > plus_ay or point < minus_ay):
                            print("Coherence error.")
                            break
                    
            if (any(point > plus_az or point < minus_az for point in segment.az)) and ("z" not in segment.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in segment.az:
                        if (point > plus_az or point < minus_az):
                            print("Coherence error.")
                            break
                    
        return number_of_errors
    
    
    ### Test to check that every segment detected initially can be found inside one of the segments found after overlapping.
    def test_no_segments_missing(self, initial_segments, final_segments):
        number_of_errors = 0
        for initial_segment in initial_segments:
            segment_found = 0
            for final_segment in final_segments:
                if initial_segment.start >= final_segment.start and initial_segment.end <= final_segment.end:
                    segment_found = 1
                    break
            else:
                if segment_found == 0:
                    number_of_errors = number_of_errors + 1
                    
        return number_of_errors
    
####################################################################################################################################
####################################################################################################################################
    '''
    This method allows to compute the max correlation and lag arrays. 
    However, this is very computationally expensive and if the amount of segments is too big, it will take a long time.
    In order to improve the performance of this process, we created the compute_corr.py file, which does the same thing but using
    the multiprocessing package to take advantage of parallel processing. This cannot be done here because of some problematic interactions
    between the multiprocessing package, IPython and Windows.
    
    In order to run compute_corr.py, just open the compute_corr.py file 
    and set up the proper paths and filenames where the acceleration data and segments are.
    Then, open a cmd at the corresponding window and write "python "compute_corr.py"". 
    The correlation and lag arrays will be exported as .npy files.
    
    signal.correlation_lags method needs Scipy 1.6.3 to work.
    '''
    ### Compute max correlation between each event of an  axis.    
    def compute_max_corr(self, segments):
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
    
    
    '''
    This method groups similar segments based on their mutual max correlation for each axis.
    To do this it takes into accounts a threshold for each axis,
    i.e. if the max correlation between 2 segment in x axis is > than threshold_ax, they are grouped together
    '''
    def group_similar_segments(self, input_segments, corr_ax, corr_ay, corr_az, threshold_ax, threshold_ay, threshold_az):
        ### Add a global index to each segment from 0 to len(segments)
        segments = copy.copy(input_segments)
        
        ### Take one segment e1, if the next one has a correlation higher than threshold, we put them into a separate list. 
        ### Repeat until there are no more segments with correlation higher than threshold for e1.
        similar_segments = []
        i = 0
        while i < len(segments):
            current_segment = copy.copy(segments[i])
            temp_similar_segments = [current_segment]
            j = i+1
            while j < len(segments):
                next_segment = copy.copy(segments[j])
                next_segment_index = next_segment.id
                
                c_ax = copy.copy(corr_ax[current_segment.id, next_segment.id])
                c_ay = copy.copy(corr_ay[current_segment.id, next_segment.id])
                c_az = copy.copy(corr_az[current_segment.id, next_segment.id])
                
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
    
    '''
    This method aligns each segment from a given group with the first segment of that group,
    using the lag where the correlation was maximum.
    '''
    def align_segments(self, similar_segments, lag_ax):    
        similar_segments_aligned = []
        for i in range(0, len(similar_segments)):
            first_segment = copy.copy(similar_segments[i][0])
            
            temp_similar_segments_aligned = []
            temp_similar_segments_aligned.append(first_segment)
            
            for j in range(1, len(similar_segments[i])):
                current_segment = copy.copy(similar_segments[i][j])
                
                lag = copy.copy(lag_ax[first_segment.id][current_segment.id])
                
                new_current_segment = copy.copy(current_segment)
                current_segment.start = int(float(new_current_segment.start)) - int(lag)
                current_segment.end = int(float(new_current_segment.end)) - int(lag)
                
                temp_similar_segments_aligned.append(current_segment)
            
            similar_segments_aligned.append(temp_similar_segments_aligned)
            
        print("Similar segments aligned.")
        return similar_segments_aligned
    
    
    ### This method removes the groups that are smaller than min_group_size.
    def remove_small_groups(self, groups, min_group_size):
        result_groups = []
        for group in groups:
            if len(group) >= min_group_size:
                result_groups.append(group)
                
        return result_groups
    
    ### Save N most common behaviors.
    def save_most_common_behaviors(self, groups, N):
        temp_groups = copy.copy(groups)
        temp_groups = sorted(temp_groups, key=lambda group: len(group))
        #temp_groups.sort(key=lambda group: len(group))

        result_groups = []
        for i in range(N):
            result_groups.append(temp_groups[len(temp_groups)-1-i])
                
        return result_groups
    
    ### Find group metrics.
    def find_group_metrics(self, groups, all_data):
        group_sizes = []
        for group in groups:
            group_sizes.append(len(group))
        print("Maximum group size: "+str(max(group_sizes)))
        print("Mean group size: "+str(np.mean(group_sizes)))
        
        ### Compute total number of segments after processing.
        number_of_segments = 0
        number_of_groups = 0
        for group in groups:
            number_of_groups = number_of_groups + 1
            for segment in group:
                number_of_segments = number_of_segments + 1
        print("Total number of segments after processing: "+str(number_of_segments))
        print("Total number of groups after processing: "+str(number_of_groups))
        
        ### Compute mean correlation coefficient (only takes account groups with size > 1)
        temp_groups = copy.copy(groups)
        corr_coefs = []
        for group in temp_groups:
            group_segment_sizes = []
            for segment in group:
                group_segment_sizes.append(len(segment.ax))
            min_segment_size = min(group_segment_sizes)
            
            group_ax = []
            for segment in group:
                segment.end = segment.start + min_segment_size
                
                [segment.setup_acceleration(data) for data in all_data if segment.filename == data.filename]
        
                group_ax.append(np.array(segment.ax))
            corr_coefs.append(np.mean(np.corrcoef(group_ax)))
               
        print("Mean correlation coefficients: "+str(np.mean(corr_coefs)))
        
        
    ### This method finds the average behavior from each of the segment groups.    
    def find_average_behavior(self, groups, mode="nanmean"):
        avrg_group_ax, avrg_group_ay, avrg_group_az, avrg_group_pressure = [], [], [], []
        if mode == "normal":
            for group in groups:
                group_ax, group_ay, group_az, group_pressure = [], [], [], []
                
                for segment in group:
                    if len(segment.ax) == 0:
                        group.remove(segment)
                
                group_min_len = min([len(segment.ax) for segment in group if len(segment.ax) > 0])
                for segment in group:
                        segment.end = segment.start + group_min_len
                        segment.ax = segment.ax[0:group_min_len]
                        segment.ay = segment.ay[0:group_min_len]
                        segment.az = segment.az[0:group_min_len]
                        segment.pressure = segment.pressure[0:group_min_len]
                        
                        group_ax.append(segment.ax)
                        group_ay.append(segment.ay)
                        group_az.append(segment.az)
                        group_pressure.append(segment.pressure[:,0])
                
                avrg_group_ax.append(np.mean(group_ax, axis = 0))
                avrg_group_ay.append(np.mean(group_ay, axis = 0))
                avrg_group_az.append(np.mean(group_az, axis = 0))
                avrg_group_pressure.append(np.mean(group_pressure, axis = 0))
                          
        if mode == "nanmean":
            temp_groups = copy.copy(groups)
            temp_groups_2 = []
            for group in temp_groups:
                temp_group = []
                group.sort(key=lambda segment: len(segment.ax))
                len_grp = len(group)
                for i in range(int(0.8*len_grp)):
                    temp_group.append(group[i])
                temp_groups_2.append(temp_group)
                    
            for group in temp_groups_2:
                group_ax, group_ay, group_az, group_pressure = [], [], [], []
                
                for segment in group:
                    group_ax.append(segment.ax)
                    group_ay.append(segment.ay)
                    group_az.append(segment.az)
                    group_pressure.append(segment.pressure[:,0])
                    
                    max_len_ax = np.array([len(array) for array in group_ax]).max()
                    max_len_ay = np.array([len(array) for array in group_ay]).max()
                    max_len_az = np.array([len(array) for array in group_az]).max()
                    max_len_pressure = np.array([len(array) for array in group_pressure]).max()
                
                group_ax = np.array([np.pad(array, (0, max_len_ax - len(array)), mode='constant', constant_values=np.nan) for array in group_ax])
                group_ay = np.array([np.pad(array, (0, max_len_ay - len(array)), mode='constant', constant_values=np.nan) for array in group_ay])
                group_az = np.array([np.pad(array, (0, max_len_az - len(array)), mode='constant', constant_values=np.nan) for array in group_az])
                group_pressure = np.array([np.pad(array, (0, max_len_pressure - len(array)), mode='constant', constant_values=np.nan) for array in group_pressure])
                
                avrg_group_ax.append(np.nanmean(group_ax, axis = 0))
                avrg_group_ay.append(np.nanmean(group_ay, axis = 0))
                avrg_group_az.append(np.nanmean(group_az, axis = 0))
                avrg_group_pressure.append(np.nanmean(group_pressure, axis = 0))
            
        return avrg_group_ax, avrg_group_ay, avrg_group_az, avrg_group_pressure

    def find_median_behavior(self, groups):
        avrg_group_ax, avrg_group_ay, avrg_group_az = [], [], []
                            
        temp_groups = copy.copy(groups)
        temp_groups_2 = []
        for group in temp_groups:
            temp_group = []
            group.sort(key=lambda segment: len(segment.ax))
            len_grp = len(group)
            for i in range(int(0.8*len_grp)):
                temp_group.append(group[i])
            temp_groups_2.append(temp_group)
                
        for group in temp_groups_2:
            group_ax, group_ay, group_az = [], [], []
            
            for segment in group:
                group_ax.append(segment.ax)
                group_ay.append(segment.ay)
                group_az.append(segment.az)
                
                max_len_ax = np.array([len(array) for array in group_ax]).max()
                max_len_ay = np.array([len(array) for array in group_ay]).max()
                max_len_az = np.array([len(array) for array in group_az]).max()
            
            group_ax = np.array([np.pad(array, (0, max_len_ax - len(array)), mode='constant', constant_values=np.nan) for array in group_ax])
            group_ay = np.array([np.pad(array, (0, max_len_ay - len(array)), mode='constant', constant_values=np.nan) for array in group_ay])
            group_az = np.array([np.pad(array, (0, max_len_az - len(array)), mode='constant', constant_values=np.nan) for array in group_az])
            
            avrg_group_ax.append(np.nanmedian(group_ax, axis = 0))
            avrg_group_ay.append(np.nanmedian(group_ay, axis = 0))
            avrg_group_az.append(np.nanmedian(group_az, axis = 0))
            
        return avrg_group_ax, avrg_group_ay, avrg_group_az
                        
                
        