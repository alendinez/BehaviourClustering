#%% Cell 1: Load data, filter signals and find segments.
import time
import copy
import warnings
import numpy as np
import data_manager
import segment_manager
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_managerager class.

### Initialize with sigma, w and mode (rest or mean).
sigma = 4
w = 50
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)
data_manager = data_manager.data_manager()
### Initialize a list to store the events from all the datasets.
all_data = []
all_segments = []

### Define the names of the datasets that we will use
filenames = ['7501394_PHAAET_rec16112018_PRincon_S1',
            '7501709_PHAAET_rec18112018_PRincon_S1',
            '7501755_PHAAET_rec27112018_PRincon_S1', 
            '8200163_PHAAET_rec14052019_PRoque_S1',
            '8200445_PHAAET_rec290422019_PRincon_S1',
            '8200473_PHAAET_rec24052019_PRincon_S2',
            '8200487_PHAAET_rec04052019_PRincon_S1',
            '8200718_PHAAET_rec08032019_PRincon',
            '8201653_PHAAET_I.Cima_rec21012021_ninho 39_36_S1',
            '8201667_PHAAET_I.Cima_rec21012021_ninho 68_21_S1',
            '8201720_PHAAET_rec31122020_ICima_ninho 71_21_S1',
            '8201959_PHAAET_rec29122020_ICima_ninho 31_36_S1']

### Detect events for a given datasets
for filename in filenames:
    
    path = ""
    
    # Load data and filter acceleration signals with a butterworth filter
    initial_data = data_manager.load_data(filename, path)
    current_data = copy.deepcopy(initial_data)
    current_data.filter_accelerations(4, 0.4)
    all_data.append(current_data)
    print("Data loaded: "+filename)
    
    '''
    ###############################
    ### Plot raw vs filtered signal
    fig, ax = plt.subplots(3,2,figsize = (8,6))
    fig.suptitle("Raw vs filtered acceleration signals")
    ax[0,0].plot(initial_data.ax[10000:10200], 'b-')
    ax[0,0].set_ylim([-3, 3])
    ax[0,1].plot(current_data.ax[10000:10200], 'b-')
    ax[0,1].set_ylim([-3, 3])
    ax[0,0].set_ylabel("ax")
    ax[1,0].plot(initial_data.ay[10000:10200], 'g-')
    ax[1,0].set_ylim([-3, 3])
    ax[1,1].plot(current_data.ay[10000:10200], 'g-')
    ax[1,1].set_ylim([-3, 3])
    ax[1,0].set_ylabel("ay")
    ax[2,0].plot(initial_data.az[10000:10200], 'r-')
    ax[2,0].set_ylim([-3, 3])
    ax[2,1].plot(current_data.az[10000:10200], 'r-')
    ax[2,1].set_ylim([-3, 3])
    ax[2,0].set_ylabel("az")
    ax[2,0].set_xlabel("Number of samples")
    ax[2,1].set_xlabel("Number of samples")
    plt.show()
    ###############################
    '''
    
    ### Find raw events for ax, ay and az.
    segments_ax = segment_manager.create_raw_segments(filename, current_data.ax, "x", "mean")
    segments_ay = segment_manager.create_raw_segments(filename, current_data.ay, "y", "mean")
    segments_az = segment_manager.create_raw_segments(filename, current_data.az, "z", "mean")
    
    ### Save initial segments into a different list to check that none of them are lost after overlapping.
    init_segments = segments_ax + segments_ay + segments_az
    print("Initial segments found: "+str(len(init_segments)))

    ### Find overlapping segments
    current_segments = copy.deepcopy(init_segments)
    current_segments = segment_manager.overlap_segments(filename, current_segments, len(current_data.ax))
    print("Segments found after overlapping: "+str(len(current_segments)))

    ### Add acceleration data and segment id to each segment.
    for segment in current_segments:
        segment.setup_acceleration(current_data)
        segment.id = current_segments.index(segment)
   
    '''
    ### Run some tests to ensure that the code has worked as expected.
    number_of_errors = segment_manager.test_tag_coherence(current_segments, current_data)
    if number_of_errors > 0:
        print("Some of the segments do not have the right axis label assigned. Number of errors: "+str(number_of_errors))
    
    number_of_errors = segment_manager.test_no_segments_missing(init_segments, current_segments)
    if number_of_errors > 0:
        print("Some of the initial segments is not inside a final segment. Number of errors: "+str(number_of_errors))
    '''
    '''
    ### Remove segments shorter than threshold.
    min_segment_size = 25
    current_segments = segment_manager.remove_short_segments(current_segments, min_segment_size)
    print("Number of segments after removing short evernts: "+str(len(current_segments)))
    '''
    ### Add segment id to each segment.
    for segment in current_segments:
        segment.id = current_segments.index(segment)
    
    ### Export segments from filename to CSV
    export_path = ""
    data_manager.export_segments(current_segments, sigma, w, filename, export_path)
    print("Segments successfully exported to .csv.")
    print("")
    
    ### Append the segments found in the current dataset into a list that contains the segments from ALL datasets.
    all_segments = all_segments + current_segments
    
    '''
    ##############################################
    ### Plot original signals vs segmented signals
    min_ax, max_ax = min(current_data.ax)-0.5, max(current_data.ax)+0.5
    min_ay, max_ay = min(current_data.ay)-0.5, max(current_data.ay)+0.5
    min_az, max_az = min(current_data.az)-0.5, max(current_data.az)+0.5
    
    std_ax = np.std(current_data.ax)
    std_ay = np.std(current_data.ay)
    std_az = np.std(current_data.az)
    
    if segment_manager.mode == "mean":
        mean_ax = np.mean(current_data.ax)
        mean_ay = np.mean(current_data.ay)
        mean_az = np.mean(current_data.az)
        
    upper_threshold_ax = mean_ax + segment_manager.sigma*std_ax
    lower_threshold_ax = mean_ax - segment_manager.sigma*std_ax
    upper_threshold_ay = mean_ay + segment_manager.sigma*std_ay
    lower_threshold_ay = mean_ay - segment_manager.sigma*std_ay
    upper_threshold_az = mean_az + segment_manager.sigma*std_az
    lower_threshold_az = mean_az - segment_manager.sigma*std_az
    
    fig, ax = plt.subplots(3,2,figsize = (16,12))
    fig.suptitle("Original signal vs segmented signal")
    ax[0,0].plot(current_data.ax, 'b-')
    ax[0,0].plot(np.full(len(current_data.ax), upper_threshold_ax), 'b-', ls=('dotted'))
    ax[0,0].plot(np.full(len(current_data.ax), lower_threshold_ax), 'b-', ls=('dotted'))
    ax[0,1].plot(np.full(len(current_data.ax), upper_threshold_ax), 'b-', ls=('dotted'))
    ax[0,1].plot(np.full(len(current_data.ax), lower_threshold_ax), 'b-', ls=('dotted'))
    ax[0,0].set_ylim([min_ax, max_ax])
    ax[0,0].set_ylabel("ax")
    ax[1,0].plot(current_data.ay, 'g-')
    ax[1,0].set_ylim([min_ay, max_ay])
    ax[1,0].plot(np.full(len(current_data.ay), upper_threshold_ay), 'g-', ls=('dotted'))
    ax[1,0].plot(np.full(len(current_data.ay), lower_threshold_ay), 'g-', ls=('dotted'))
    ax[1,1].plot(np.full(len(current_data.ay), upper_threshold_ay), 'g-', ls=('dotted'))
    ax[1,1].plot(np.full(len(current_data.ay), lower_threshold_ay), 'g-', ls=('dotted'))
    ax[1,0].set_ylabel("ay")
    ax[2,0].plot(current_data.az, 'r-')
    ax[2,0].plot(np.full(len(current_data.az), upper_threshold_az), 'r-', ls=('dotted'))
    ax[2,0].plot(np.full(len(current_data.az), lower_threshold_az), 'r-', ls=('dotted'))
    ax[2,1].plot(np.full(len(current_data.az), upper_threshold_az), 'r-', ls=('dotted'))
    ax[2,1].plot(np.full(len(current_data.az), lower_threshold_az), 'r-', ls=('dotted'))
    ax[2,0].set_ylim([min_az, max_az])
    ax[2,0].set_ylabel("az")
    
    segments_ax = np.empty(len(current_data.ax))
    segments_ax[:] = np.nan
    segments_ay = np.empty(len(current_data.ay))
    segments_ay[:] = np.nan
    segments_az = np.empty(len(current_data.az))
    segments_az[:] = np.nan
    
    for segment in current_segments:
        segments_ax[int(segment.start):int(segment.end)] = current_data.ax[int(segment.start):int(segment.end)]
        segments_ay[int(segment.start):int(segment.end)] = current_data.ay[int(segment.start):int(segment.end)]
        segments_az[int(segment.start):int(segment.end)] = current_data.az[int(segment.start):int(segment.end)]
        
    ax[0,1].plot(segments_ax, 'b-')
    ax[0,1].set_ylim([min_ax, max_ax])
    ax[1,1].plot(segments_ay, 'g-')
    ax[1,1].set_ylim([min_ay, max_ay])
    ax[2,1].plot(segments_az, 'r-')
    ax[2,1].set_ylim([min_az, max_az])

    ax[2,0].set_xlabel("Number of samples")
    ax[2,1].set_xlabel("Number of samples")
    plt.show()
    ##############################################
    ##############################################
    '''
i = 0
for segment in all_segments:
    segment.id = i
    i = i+1

### Export all segments to CSV
export_path = ""

data_manager.export_all_segments(all_segments, sigma, w, export_path)
print("All segments successfully exported to .csv.")
print("")
   
print("Total number of segments: "+str(len(all_segments)))
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 2: Find some metrics for the detected segments.

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

#%% Cell 2.1: Plot segment length histogram
from matplotlib import pyplot as plt
import numpy as np

segment_lengths = []                
for segment in all_segments:
    if len(segment.ax) < 2000:
        segment_lengths.append(len(segment.ax))

### Plot segment length histogram
hist, bin_edges = np.histogram(segment_lengths)

fig, ax = plt.subplots(1,1,figsize = (8,6))
ax.title.set_text("Event length histogram.")
ax.hist(segment_lengths, bins=200, log=True)  # arguments are passed to np.histogram)

#%% Cell 3: Set up some values for plotting.
import numpy as np

### Set upper and lower thresholds for plotting
for segmentdata in all_data:
    std_ax = np.std(segmentdata.ax)
    std_ay = np.std(segmentdata.ay)
    std_az = np.std(segmentdata.az)
    
    if segment_manager.mode == "mean":
        mean_ax = np.mean(segmentdata.ax)
        mean_ay = np.mean(segmentdata.ay)
        mean_az = np.mean(segmentdata.az)
        
        upper_threshold_ax = mean_ax + segment_manager.sigma*std_ax
        lower_threshold_ax = mean_ax - segment_manager.sigma*std_ax
        upper_threshold_ay = mean_ay + segment_manager.sigma*std_ay
        lower_threshold_ay = mean_ay - segment_manager.sigma*std_ay
        upper_threshold_az = mean_az + segment_manager.sigma*std_az
        lower_threshold_az = mean_az - segment_manager.sigma*std_az
        
    if segment_manager.mode == "rest":
        upper_threshold_ax = 0 + segment_manager.sigma*std_ax
        lower_threshold_ax = 0 - segment_manager.sigma*std_ax
        upper_threshold_ay = 0 + segment_manager.sigma*std_ay
        lower_threshold_ay = 0 - segment_manager.sigma*std_ay
        upper_threshold_az = 0 + segment_manager.sigma*std_az
        lower_threshold_az = 0 - segment_manager.sigma*std_az
        
    for segment in all_segments:
        if segment.filename == segmentdata.filename:
            segment.setup_thresholds(upper_threshold_ax, lower_threshold_ax, upper_threshold_ay, lower_threshold_ay, upper_threshold_az, lower_threshold_az)

### Set min and max values of acceleration for axis scaling in plotting
for data in all_data:
    min_ax, max_ax = min(data.ax)-1, max(data.ax)+1
    min_ay, max_ay = min(data.ay)-1, max(data.ay)+1
    min_az, max_az = min(data.az)-1, max(data.az)+1
    min_pressure, max_pressure = min(data.pressure)-10, max(data.pressure)+10
    for segment in all_segments:
        if segment.filename == data.filename:
            segment.min_ax, segment.max_ax = min_ax, max_ax
            segment.min_ay, segment.max_ay = min_ay, max_ay
            segment.min_az, segment.max_az = min_az, max_az
            segment.min_pressure, segment.max_pressure = min_pressure, max_pressure

#%% Cell 3.1: Plot some of the segments.
from matplotlib import pyplot as plt
 
### Plot segments given a condition
j = 0
while j <= 5:
    for segment in all_segments:
        if len(segment.axis) == 3:
            j += 1
            fig, ax = plt.subplots(4,1,figsize = (8,6))
            ax[0].title.set_text("Segment id: "+str(segment.id)+". Segment Axis: "+segment.axis)
            ax[0].plot(segment.ax, 'b-')
            ax[0].plot(np.full(len(segment.ax), segment.upper_threshold_ax), 'b-', ls=('dotted'))
            ax[0].plot(np.full(len(segment.ax), segment.lower_threshold_ax), 'b-', ls=('dotted'))
            ax[0].plot(np.full(len(segment.ax), 0), 'b-', ls=('dashed'), lw=0.5)
            ax[0].set_ylim([segment.min_ax, segment.max_ax])
            ax[0].set_ylabel("ax")
            ax[1].plot(segment.ay, 'g-')
            ax[1].plot(np.full(len(segment.ay), segment.upper_threshold_ay), 'g-', ls=('dotted'))
            ax[1].plot(np.full(len(segment.ay), segment.lower_threshold_ay), 'g-', ls=('dotted'))
            ax[1].plot(np.full(len(segment.ay), 0), 'g-', ls=('dashed'), lw=0.5)
            ax[1].set_ylim([segment.min_ay, segment.max_ay])
            ax[1].set_ylabel("ay")
            ax[2].plot(segment.az, 'r-')
            ax[2].plot(np.full(len(segment.az), segment.upper_threshold_az), 'r-', ls=('dotted'))
            ax[2].plot(np.full(len(segment.az), segment.lower_threshold_az), 'r-', ls=('dotted'))
            ax[2].plot(np.full(len(segment.ay), 0), 'g-', ls=('dashed'), lw=0.5)
            ax[2].set_ylim([segment.min_az, segment.max_az])
            ax[2].set_ylabel("az")
            ax[3].plot(segment.pressure, 'k-')
            ax[3].set_ylim([segment.min_pressure, segment.max_pressure])
            ax[3].set_xlabel("number of samples")
            ax[3].set_ylabel("pressure (mBar)")
            plt.show()


#%% Cell 4: Compute max correlation between each segment.
import os
import numpy as np
import segment_manager
import time

'''
This cell allows to compute the max correlation and lag arrays. 
However, this is very computationally expensive and if the amount of segments is too big, it will take a long time.
In order to improve the performance of this process, we created the compute_corr.py file, which does the same thing but using
the multiprocessing package to take advantage of parallel processing. This cannot be done here because of some problematic interactions
between the multiprocessing package, IPython and Windows.

In order to run compute_corr.py, just open the compute_corr.py file 
and set up the proper paths and filenames where the acceleration data and segments are.
Then, open a cmd at the corresponding window and write "python "compute_corr.py"". 
The correlation and lag arrays will be exported as .npy files.
'''

start_time = time.time()
   
sigma = 6
w = 50
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)

temp_segments = copy.deepcopy(all_segments)

i = 0
segments_ax, segments_ay, segments_az = [], [], []
for segment in temp_segments:
    segment.id = i
    segments_ax.append(np.array(segment.ax))
    segments_ay.append(np.array(segment.ay))
    segments_az.append(np.array(segment.az))
    i = i + 1

maxcorr_ax, lag_ax = segment_manager.compute_max_corr(segments_ax)
maxcorr_ay, lag_ay = segment_manager.compute_max_corr(segments_ay)
maxcorr_az, lag_az = segment_manager.compute_max_corr(segments_az)

### Save correlation and lag into numpy format
path = ""
np.save(os.path.join(path, 'maxcorr_ax.npy'), maxcorr_ax)
np.save(os.path.join(path, 'maxcorr_ay.npy'), maxcorr_ay)
np.save(os.path.join(path, 'maxcorr_az.npy'), maxcorr_az)
np.save(os.path.join(path, 'lag_ax.npy'), lag_ax)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")


#%% Cell 5: Load events, correlation matrix and lag matrix and plot max correlation matrix for each axis
import numpy as np
import data_manager
from matplotlib import pyplot as plt

'''
This cell allows to load the previously created segments and the correlation and lag arrays without the need of running the pipeline again.
'''
data_manager = data_manager.data_manager()
### Load acceleration data
sigma = 4
w = 50
filenames = ['7501394_PHAAET_rec16112018_PRincon_S1',
            '7501709_PHAAET_rec18112018_PRincon_S1',
            '7501755_PHAAET_rec27112018_PRincon_S1', 
            '8200163_PHAAET_rec14052019_PRoque_S1',
            '8200445_PHAAET_rec290422019_PRincon_S1',
            '8200473_PHAAET_rec24052019_PRincon_S2',
            '8200487_PHAAET_rec04052019_PRincon_S1',
            '8200718_PHAAET_rec08032019_PRincon',
            '8201653_PHAAET_I.Cima_rec21012021_ninho 39_36_S1',
            '8201667_PHAAET_I.Cima_rec21012021_ninho 68_21_S1',
            '8201720_PHAAET_rec31122020_ICima_ninho 71_21_S1',
            '8201959_PHAAET_rec29122020_ICima_ninho 31_36_S1']

all_data = []
for filename in filenames:
    datapath =''+filename+'\\'   
    # Load data and filter acceleration signals with a butterworth filter
    data = data_manager.load_data(filename, datapath)
    data.filter_accelerations(4, 0.4)
    all_data.append(data)
    print("Data loaded: "+filename)

### Load previously created segments
path = ""
all_segments = data_manager.load_all_segments(path, sigma, w)
for data in all_data:
        for segment in all_segments:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)    

### Load correlation and lag arrays            
maxcorr_ax = np.load(path+"maxcorr_ax.npy")
maxcorr_ay = np.load(path+"maxcorr_ay.npy")
maxcorr_az = np.load(path+"maxcorr_az.npy")

### Plot the max correlation arrays
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,figsize = (18,9))
im = ax[0].imshow(maxcorr_ax)
ax[0].set_xlabel("ax segment index")
ax[0].set_ylabel("ax segment index")
ax[1].imshow(maxcorr_ay)
ax[1].set_xlabel("ay segment index")
ax[1].set_ylabel("ay segment index")
ax[2].imshow(maxcorr_az)
ax[2].set_xlabel("az segment index")
ax[2].set_ylabel("az segment index")
fig.colorbar(im, ax=ax.ravel().tolist(), orientation = 'horizontal', aspect = 40)
fig.suptitle('Max correlation between each segment', y = 0.85)
plt.show()

#%% Cell 5.1: Load data and save it as npy file to save time in next steps.
import os
import numpy as np
import data_manager

data_manager = data_manager.data_manager()
### Load acceleration data
sigma = 10
w = 50
filenames = ['7501394_PHAAET_rec16112018_PRincon_S1',
            '7501709_PHAAET_rec18112018_PRincon_S1',
            '7501755_PHAAET_rec27112018_PRincon_S1', 
            '8200163_PHAAET_rec14052019_PRoque_S1',
            '8200445_PHAAET_rec290422019_PRincon_S1',
            '8200473_PHAAET_rec24052019_PRincon_S2',
            '8200487_PHAAET_rec04052019_PRincon_S1',
            '8200718_PHAAET_rec08032019_PRincon',
            '8201653_PHAAET_I.Cima_rec21012021_ninho 39_36_S1',
            '8201667_PHAAET_I.Cima_rec21012021_ninho 68_21_S1',
            '8201720_PHAAET_rec31122020_ICima_ninho 71_21_S1',
            '8201959_PHAAET_rec29122020_ICima_ninho 31_36_S1']

all_data = []
for filename in filenames:
    datapath =''+filename+'\\'   
    # Load data and filter acceleration signals with a butterworth filter
    data = data_manager.load_data(filename, datapath)
    data.filter_accelerations(4, 0.4)
    all_data.append(data)
    print("Data loaded: "+filename)

path = ""
np.save(os.path.join(path, 'all_data.npy'), all_data)

#%% Cell 6: Group segments based on max correlation, remove groups smaller than a threshold
### and align the segments from each group. Compute some group metrics and plot the average behaviour from each group.
import copy
import time
import numpy as np
import segment_manager
start_time = time.time()

'''
If the number of segments is too big, it is recommended to create the groups by running "group_segments.py" from cmd. 
If the number of segments is small (e.g smaller than 15k segments) and you want to create the groups here, uncomment the next block of code.
'''

'''
### Group segments
segment_manager = segment_manager.segment_manager(sigma, w)
threshold_ax = 0.3
threshold_ay = 0
threshold_az = 0.3
input_segments = copy.copy(all_segments)
groups_raw = segment_manager.group_similar_segments(input_segments, maxcorr_ax, maxcorr_ay, maxcorr_az, threshold_ax, threshold_ay, threshold_az)
'''

segment_manager = segment_manager.segment_manager(4, 50)
path = ""
all_data = np.load(path+"all_data.npy", allow_pickle = True)
lag_ax = np.load(path+"lag_ax.npy")

path2 = ""
groups_raw = np.load(path2+"groups_raw.npy", allow_pickle = True)
    
'''
### Discard smaller groups
min_group_size = 30
groups = segment_manager.remove_small_groups(groups_raw, min_group_size)
'''

### Save N most common behaviors
N = 7
groups = segment_manager.save_most_common_behaviors(groups_raw, N)

### Align segments from the same group
groups = segment_manager.align_segments(groups, lag_ax)
### Set up acceleration for the aligned segments
for data in all_data:
    for group in groups:
        for segment in group:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)

### Find some group metrics
#segment_manager.find_group_metrics(groups, all_data)

### Find average behavior for each group in the three axis and plot it
avrg_group_ax, avrg_group_ay, avrg_group_az, avrg_group_pressure = segment_manager.find_average_behavior(groups, mode="nanmean")

### Add a group label to each segment and save every segment into a common list again
group_label = 0
input_segments = []
for group in groups:
    for segment in group:
        segment.group_label = group_label
        input_segments.append(segment)
    group_label = group_label + 1
    
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 6.1: Check the number of groups with more than 100 elements and
# the percentage of segments that end up in the smaller groups with less than 100 elements.
import copy
import time
import numpy as np
import segment_manager
start_time = time.time()

'''
If the number of segments is too big, it is recommended to create the groups by running "group_segments.py" from cmd. 
If the number of segments is small (e.g smaller than 15k segments) and you want to create the groups here, uncomment the next block of code.
'''

'''
### Group segments
segment_manager = segment_manager.segment_manager(sigma, w)
threshold_ax = 0.3
threshold_ay = 0
threshold_az = 0.3
input_segments = copy.copy(all_segments)
groups_raw = segment_manager.group_similar_segments(input_segments, maxcorr_ax, maxcorr_ay, maxcorr_az, threshold_ax, threshold_ay, threshold_az)
'''

segment_manager = segment_manager.segment_manager(4, 50)
path = ""

path2 = ""
groups_raw = np.load(path2+"groups_raw.npy", allow_pickle = True)
    

plus100 = 0
total_segments = 0
for group in groups_raw:
    if len(group) >= 100:
        plus100 = plus100+1
        total_segments = total_segments+len(group)
        
percentage = 100-(total_segments/20813)*100
print("Number of groups with moer than 100 elements: "+str(plus100))
print("Percentage of segments out of these groups: "+str(percentage))
    
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

    
#%% Cell 7: Plot all the segments and average behavior from each group.
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

def get_cmap(n, name='YlOrRd'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None,):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap("my_cmap", segmentdata=cdict, N=256)
    return cmp

for group in groups:
    group = sorted(group, key=lambda segment: len(segment.ax))

hex_list1 = ['#005e97', '#176f9e', '#3080a4', '#4890a9', '#60a1ad', '#78b1b1', '#92c1b4', '#acd1b7', '#c7e0b9', '#e2f0ba', '#ffffbb', '#e2f0ba', '#c7e0b9', '#acd1b7', '#92c1b4', '#78b1b1', '#60a1ad', '#4890a9', '#3080a4', '#176f9e', '#005e97']
hex_list2 = ['#d90000', '#e23713', '#ea5325', '#f26b36', '#f88148', '#fd975a', '#ffac6c', '#ffc280', '#ffd794', '#ffeba7', '#ffffbb', '#ffeba7', '#ffd794', '#ffc280', '#ffac6c', '#fd975a', '#f88148', '#f26b36', '#ea5325', '#e23713', '#d90000']
hex_list3 = ['#004a00', '#185c06', '#306e14', '#478025', '#5f9237', '#78a44a', '#92b65e', '#acc874', '#c7da8b', '#e3eda2', '#ffffbb', '#e3eda2', '#c7da8b', '#acc874', '#92b65e', '#78a44a', '#5f9237', '#478025', '#306e14', '#185c06', '#004a00']
num_segments = 10000

for i in range(len(groups)):
    cmap11 = get_continuous_cmap(hex_list1)
    cmap22 = get_continuous_cmap(hex_list2)
    cmap33 = get_continuous_cmap(hex_list3)
    
    plt.cm.register_cmap("cmap1", cmap11)
    plt.cm.register_cmap("cmap2", cmap22)
    plt.cm.register_cmap("cmap3", cmap33)
    
    cmap1 = get_cmap(len(groups[i]), "cmap1")
    cmap2 = get_cmap(len(groups[i]), "cmap2")
    cmap3 = get_cmap(len(groups[i]), "cmap3")
    
    max_ax, min_ax = max(avrg_group_ax[i]), min(avrg_group_ax[i])
    max_ay, min_ay = max(avrg_group_ay[i]), min(avrg_group_ay[i])
    max_az, min_az = max(avrg_group_az[i]), min(avrg_group_az[i])
    
    
    fig, ax = plt.subplots(3,2,figsize = (16,12))
    ax[0,1].plot(avrg_group_ax[i], 'lightseagreen')
    ax[0,1].set_ylim([min_ax-1, max_ax+1])
    ax[1,1].plot(avrg_group_ay[i], 'coral')
    ax[1,1].set_ylim([min_ay-1, max_ay+1])
    ax[2,1].plot(avrg_group_az[i], 'olive')
    ax[2,1].set_ylim([min_az-1, max_az+1])
    
    j = 0
    for segment in groups[i]:
        ax[0,0].plot(segment.ax[:len(avrg_group_ax[i])], c=cmap1(j), lw=0.3, alpha = 0.3)
        ax[0,0].set_ylim([-9, 9])
        ax[0,0].set_ylabel("ax")
        ax[1,0].plot(segment.ay[:len(avrg_group_ax[i])], c=cmap2(j), lw=0.3, alpha = 0.3)
        ax[1,0].set_ylim([-9, 9])
        ax[1,0].set_ylabel("ay")
        ax[2,0].plot(segment.az[:len(avrg_group_ax[i])], c=cmap3(j), lw=0.3, alpha = 0.3)
        ax[2,0].set_ylim([-9, 9])
        ax[2,0].set_ylabel("az")
        
        j += 1
    
    fig.suptitle(f'All segments and avrg segment from group {i}, group size: {str(len(groups[i]))}', y = 0.9)
    plt.show()
    
#%% Cell 8: Cross-validation 1. Create train and test data for Reservoir Computing (80% train, 20% test).
import copy
import tsaug
import random
import numpy as np
from tsaug.visualization import plot
from matplotlib import pyplot as plt

temp_groups = copy.deepcopy(groups)
#temp_groups = groups
k = 0
for group in temp_groups:
    for segment in group:
        segment.group_label = k
    k += 1
    
segments_train, segments_test = [], []

num_groups = len(groups)
max_group_length = int(max([len(group) for group in groups[1:len(groups)-1]]))
#max_group_length = len(groups[0])

num_segments_train_pergroup = int(0.8*max_group_length)
num_segments_test_pergroup = max_group_length - num_segments_train_pergroup
num_segments_train = num_segments_train_pergroup*num_groups
num_segments_test = num_segments_test_pergroup*num_groups  # Number of segments from each group that we will use to train the network.

train_segments = []
test_segments = []
train_data_ax, train_data_ay, train_data_az, len_segments_train, labels_train = [], [], [], [], []
test_data_ax, test_data_ay, test_data_az, len_segments_test, labels_test = [], [], [], [], []          

### Perform data augmentation in order to have the same number of examples from each behavioral group.
for group in temp_groups:
    while len(group) < max_group_length:
        #print(len(group))
        try:
            current_segment = copy.copy(random.choice(group))
            
            csa_ax = tsaug.AddNoise(scale=0.025).augment(current_segment.ax)
            csa_ay = tsaug.AddNoise(scale=0.025).augment(current_segment.ay)
            csa_az = tsaug.AddNoise(scale=0.025).augment(current_segment.az)
                
            current_segment.ax, current_segment.ay, current_segment.az = csa_ax, csa_ay, csa_az
            group.append(current_segment)
        except:
            continue
        ''' 
        fig, ax = plt.subplots(2,1,figsize = (8,6))
        ax[0].plot(current_segment.ax)
        ax[1].plot(csa_ax)
        plt.show()
        '''

for i in range(0, num_segments_train_pergroup):
    for group in temp_groups:
        current_segment = random.choice(group)
        segments_train.append(current_segment)
        len_segments_train.append(len(current_segment.ax))
        for ax in current_segment.ax:
            train_data_ax.append(ax)
        for ay in current_segment.ay:
            train_data_ay.append(ay)
        for az in current_segment.az:
            train_data_az.append(az)
        labels_train.append(current_segment.group_label)
        group.remove(current_segment)
            
   
for i in range(0, num_segments_test_pergroup):
    for group in temp_groups:
        current_segment = random.choice(group)
        segments_test.append(current_segment)
random.shuffle(segments_test)

for current_segment in segments_test:
    len_segments_test.append(len(current_segment.ax))
    for ax in current_segment.ax:
        test_data_ax.append(ax)
    for ay in current_segment.ay:
        test_data_ay.append(ay)
    for az in current_segment.az:
        test_data_az.append(az)
    labels_test.append(current_segment.group_label)
      
labels_train, labels_test = np.array(labels_train), np.array(labels_test)
train_data = np.array([train_data_ax, train_data_ay, train_data_az])
test_data = np.array([test_data_ax, test_data_ay, test_data_az])    
        
#%% Cell 8.1: Cross-validation 2. Create train and test data for Reservoir Computing (Leave one out).
import copy
import tsaug
import random
import numpy as np
from matplotlib import pyplot as plt


temp_groups = copy.deepcopy(groups)
#temp_groups = groups
k = 0
for group in temp_groups:
    for segment in group:
        segment.group_label = k
    k += 1
    
num_groups = len(groups)
#max_group_length = int(max([len(group) for group in groups[1:len(groups)-1]]))
max_group_length = 3500

num_segments_train_pergroup = max_group_length
num_segments_train = num_segments_train_pergroup*num_groups  # Number of segments from each group that we will use to train the network.

train_segments = []
test_segments = []
train_data_ax, train_data_ay, train_data_az, len_segments_train, labels_train = [], [], [], [], []
test_data_ax, test_data_ay, test_data_az, len_segments_test, labels_test = [], [], [], [], []          

test_filename = "8201720_PHAAET_rec31122020_ICima_ninho 71_21_S1"

for group in temp_groups:
    for current_segment in group:
        if current_segment.filename == test_filename:
            test_segments.append(current_segment)
            group.remove(current_segment)

random.shuffle(test_segments)
for current_segment in test_segments:
    len_segments_test.append(len(current_segment.ax))
    for ax in current_segment.ax:
        test_data_ax.append(ax)
    for ay in current_segment.ay:
        test_data_ay.append(ay)
    for az in current_segment.az:
        test_data_az.append(az)
    labels_test.append(current_segment.group_label)

for group in temp_groups:
    while len(group) < max_group_length:
        #print(len(group))
        try:
            current_segment = copy.copy(random.choice(group))
            
            rand = random.uniform(0, 1)
            if rand >= 0.5:
                csa_ax = tsaug.AddNoise(scale=0.02).augment(current_segment.ax)
                csa_ay = tsaug.AddNoise(scale=0.02).augment(current_segment.ay)
                csa_az = tsaug.AddNoise(scale=0.02).augment(current_segment.az)
            else:
                csa_ax = tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(current_segment.ax)
                csa_ay = tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(current_segment.ay)
                csa_az = tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(current_segment.az)
                
            current_segment.ax, current_segment.ay, current_segment.az = csa_ax, csa_ay, csa_az
            group.append(current_segment)
        except:
            continue
        
for i in range(0, num_segments_train_pergroup):
    for group in temp_groups:
        current_segment = random.choice(group)
        train_segments.append(current_segment)
        len_segments_train.append(len(current_segment.ax))
        for ax in current_segment.ax:
            train_data_ax.append(ax)
        for ay in current_segment.ay:
            train_data_ay.append(ay)
        for az in current_segment.az:
            train_data_az.append(az)
        labels_train.append(current_segment.group_label)        

num_segments_train = len(train_segments)
num_segments_test = len(test_segments)

labels_train, labels_test = np.array(labels_train), np.array(labels_test)
train_data = np.array([train_data_ax, train_data_ay, train_data_az])
test_data = np.array([test_data_ax, test_data_ay, test_data_az])         

#%% Cell 9: Train and test Reservoir Computer Network using previously generated segments as input data.
import copy
import network as Network

Network = Network.Network()
num_nodes = 100

input_probability = 0.3
reservoir_probability = 0.3
classifier = "log"

Network.T = sum(len_segments_train)  
Network.n_min = 1
Network.K = 3
Network.N = num_nodes

Network.setup_network(train_data, num_nodes, input_probability, reservoir_probability, num_groups, num_segments_train)
Network.train_network(num_groups, classifier, num_segments_train, len_segments_train, labels_train, num_nodes)

Network.mean_test_matrix = np.zeros([Network.N, num_segments_test])
Network.test_network(test_data, num_segments_test, len_segments_test, num_nodes, num_groups, sum(len_segments_test))

if classifier == 'log':
    print(f'Performance using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,labels_test.T)}')
    prediction = Network.regressor.predict(Network.mean_test_matrix.T)
    
    
#%% Cell 10: Plot confusion matrix.
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize = (12,9))
disp = plot_confusion_matrix(Network.regressor, Network.mean_test_matrix.T, labels_test.T, normalize='true', ax=ax)
disp.ax_.set_title("Confusion matrix")
plt.show()

#%% Check how many segments from each axis do we have
path = ""
all_segments = data_manager.load_all_segments(path, 6, 50)

num_x, num_y, num_z, num_xy, num_xz, num_yz, num_xyz = 0, 0, 0, 0, 0, 0, 0

for segment in all_segments:
    if segment.axis == "x":
        num_x = num_x + 1
    if segment.axis == "y":
        num_y = num_y + 1
    if segment.axis == "z":
        num_z = num_z + 1
    if segment.axis == "xy":
        num_xy = num_xy + 1
    if segment.axis == "xz":
        num_xz = num_xz + 1
    if segment.axis == "yz":
        num_yz = num_yz + 1
    if segment.axis == "xyz":
        num_xyz = num_xyz + 1

data = [num_x, num_y, num_z, num_xy, num_xz, num_yz, num_xyz]
index = ["x", "y", "z", "xy", "xz", "yz", "xyz"]
colors = ["firebrick", "orangered", "tomato", "salmon", "coral", "lightsalmon", "peachpuff"]

fig, ax = plt.subplots(1,1,figsize = (12,9))
fig.suptitle("Number of segments for each axis or axis group.")
ax.bar(index, data, color = colors)
ax.set_ylabel("Number of segments")
ax.set_xlabel("Axis")
plt.show()
