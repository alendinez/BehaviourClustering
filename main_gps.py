#%% Cell 1: Load data, filter signals and find segments.
import os
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
sigma = 3
w = 50
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)
data_manager = data_manager.data_manager()
### Initialize a list to store the events from all the datasets.
all_data = []
all_segments = []

### Define the names of the datasets that we will use
filenames = ['8201959_PHAAET_rec29122020_ICima_ninho 31_36_S1']
### Detect events for a given datasets
for filename in filenames:
    
    path =''+filename+'\\'
    
    # Load data and filter acceleration signals with a butterworth filter
    initial_data = data_manager.load_data_gps_timestamp(filename, path)
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
        #segment.setup_gps_data(current_data)
        #segment.setup_timestamp(current_data)
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
        
    all_segments = all_segments + current_segments

output_segments = []
for segment in all_segments:
    if len(segment.axis) > 0:
        output_segments.append(segment)


i = 0
for segment in output_segments:
    segment.id = i
    i += 1
    
path_gps = ""
np.save(os.path.join(path_gps, 'allsegments_gps.npy'), output_segments)
np.save(os.path.join(path_gps, 'alldata_gps.npy'), all_data)        

print("Total number of segments: "+str(len(output_segments)))
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
import time
import numpy as np
import segment_manager

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
   
sigma = 4
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

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 5: Group segments based on max correlation, remove groups smaller than a threshold
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
segment_manager = segment_manager.segment_manager(8, 50)
path = ""
all_data = np.load(path+"alldata_gps.npy", allow_pickle = True)
lag_ax = np.load(path+"lag_ax.npy")

groups_raw = np.load(path+"groups_raw.npy", allow_pickle = True)

### Save N most common behaviors
N = 30
groups = segment_manager.save_most_common_behaviors(groups_raw, N)

### Align segments from the same group
groups = segment_manager.align_segments(groups, lag_ax)
### Set up acceleration for the aligned segments
for data in all_data:
    for group in groups:
        for segment in group:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
                segment.setup_gps_data(data)
                #segment.setup_timestamp(data)

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

#%% Cell 6: Plot all the segments and average behavior from each group.
from matplotlib import pyplot as plt

def get_cmap(n, name='YlOrRd'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

num_segments = 10000
for i in range(len(groups)):
    cmap1 = get_cmap(len(groups[i]), "YlGnBu")
    cmap2 = get_cmap(len(groups[i]), "PuBuGn")
    cmap3 = get_cmap(len(groups[i]), "YlOrRd")
    cmap4 = get_cmap(len(groups[i]), "PuRd")
    
    max_ax, min_ax = max(avrg_group_ax[i]), min(avrg_group_ax[i])
    max_ay, min_ay = max(avrg_group_ay[i]), min(avrg_group_ay[i])
    max_az, min_az = max(avrg_group_az[i]), min(avrg_group_az[i])
    max_pressure, min_pressure = max(avrg_group_pressure[i]), min(avrg_group_pressure[i])
    
    
    fig, ax = plt.subplots(4,2,figsize = (16,12))
    ax[0,1].plot(avrg_group_ax[i], 'lightseagreen')
    ax[0,1].set_ylim([min_ax-1, max_ax+1])
    ax[1,1].plot(avrg_group_ay[i], 'coral')
    ax[1,1].set_ylim([min_ay-1, max_ay+1])
    ax[2,1].plot(avrg_group_az[i], 'olive')
    ax[2,1].set_ylim([min_az-1, max_az+1])
    
    ax[3,1].plot(avrg_group_pressure[i], 'sienna')
    ax[3,1].set_ylim([min_pressure-10, max_pressure+10])
    ax[3,1].set_xlabel('number of samples')
    
    
    j = 0
    for segment in groups[i]:
        ax[0,0].plot(segment.ax[:len(avrg_group_ax[i])], c='lightseagreen', lw=1, alpha = 0.3)
        ax[0,0].set_ylim([-9, 9])
        ax[0,0].set_ylabel("ax")
        ax[1,0].plot(segment.ay[:len(avrg_group_ax[i])], c='coral', lw=1, alpha = 0.3)
        ax[1,0].set_ylim([-9, 9])
        ax[1,0].set_ylabel("ay")
        ax[2,0].plot(segment.az[:len(avrg_group_ax[i])], c='olive', lw=1, alpha = 0.3)
        ax[2,0].set_ylim([-9, 9])
        ax[2,0].set_ylabel("az")
        
        
        ax[3,0].plot(segment.pressure[:len(avrg_group_ax[i])], c='sienna', lw=1, alpha = 0.3)
        ax[3,0].set_ylim([950, 1250])
        ax[3,0].set_ylabel("pressure")
        ax[3,0].set_xlabel('number of samples')
        
        j += 1
    
    fig.suptitle(f'All segments and avrg segment from group {i}, group size: {str(len(groups[i]))}', y = 0.9)
    plt.show()


#%% Plot GPS data with cartopy library
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import data_manager
import time
import numpy as np

start_time = time.time()

data_manager = data_manager.data_manager()

path = ""
all_data = np.load(path+"alldata_gps.npy", allow_pickle = True)
all_data[0].longitude, all_data[0].latitude = data_manager.interpolate_gps_data(all_data[0])

current_groups = [groups[0], groups[9]]

for data in all_data:
    for group in current_groups:
        for segment in group:
            if segment.filename == data.filename:
                segment.setup_gps_data(data)

max_lats, max_longs = [], []
min_lats, min_longs = [], []

for data in all_data:
    max_lats.append(max(data.latitude[np.nonzero(data.latitude)]))
    min_lats.append(min(data.latitude[np.nonzero(data.latitude)]))
    max_longs.append(max(data.longitude[np.nonzero(data.longitude)]))
    min_longs.append(min(data.longitude[np.nonzero(data.longitude)]))
    
min_lat, min_long = min(min_lats)-0.25, min(min_longs)-0.25
max_lat, max_long = max(max_lats)+0.25, max(max_longs)+0.25

# Create a Stamen terrain background instance.
stamen_terrain = cimgt.Stamen('terrain')

fig = plt.figure(figsize = (32,24))

# Create a GeoAxes in the tile's projection.
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

#BBox = [-28.037, -20.786, 12.715, 17.545]
BBox = [min_long, max_long, min_lat, max_lat]
colors = ["gold", "orange", "darkolivegreen", "limegreen"]     

#BBox = [-24.7331, -24.5983, 14.9359, 15.0114]
# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent(BBox, crs=ccrs.PlateCarree())

# Add the Stamen data at zoom level 13.
ax.add_image(stamen_terrain, 8)

for i in range(len(all_data)):
    latitude = all_data[i].latitude[np.nonzero(all_data[i].latitude)]
    longitude = all_data[i].longitude[np.nonzero(all_data[i].longitude)]
    #plt.plot(longitude, latitude, c = colors[i])
    plt.plot(longitude, latitude,
         color='k', ls=':',
         transform=ccrs.PlateCarree())
    
    i = 0
    for group in current_groups:
        for segment in group:
            lat = segment.latitude[0]
            lon = segment.longitude[0]
            size = len(segment.ax)
            plt.scatter(lon, lat, s=size*2, color=colors[i],transform=ccrs.PlateCarree())
        i = i+1
            
plt.show()

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")


#%% Plot timestamp histogram
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import data_manager
import time
import numpy as np

start_time = time.time()

segment_hours = []
current_group = groups[9]

for data in all_data:
    for segment in current_group:
        if segment.filename == data.filename:
            segment.setup_timestamp(data)

for segment in current_group:
    segment_hours.append(segment.timestamp[0].hour)
    
### Plot segment length histogram
bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
hist, bin_edges = np.histogram(segment_hours)

fig, ax = plt.subplots(1,1,figsize = (8,6))
ax.title.set_text("Timestamp histogram of flying group 9")
ax.hist(segment_hours, bins=bins, log=False, color = 'lightseagreen')
ax.set_ylabel('Number of segments')
ax.grid(True)
ax.set_xlabel('Hours')
plt.xticks(bins)
    
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")


#%%% Plot GPS data with folium library
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import data_manager
import time
import numpy as np
import folium
from scipy import signal

start_time = time.time()


data_manager = data_manager.data_manager()

path = ""
all_data = np.load(path+"alldata_gps.npy", allow_pickle = True)
all_data[0].longitude, all_data[0].latitude = data_manager.interpolate_gps_data(all_data[0])

current_groups = [groups[0], groups[9]]

for data in all_data:
    for group in current_groups:
        for segment in group:
            if segment.filename == data.filename:
                segment.setup_gps_data(data)

# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
max_lats, max_longs = [], []
min_lats, min_longs = [], []

for data in all_data:
    max_lats.append(max(data.latitude[np.nonzero(data.latitude)]))
    min_lats.append(min(data.latitude[np.nonzero(data.latitude)]))
    max_longs.append(max(data.longitude[np.nonzero(data.longitude)]))
    min_longs.append(min(data.longitude[np.nonzero(data.longitude)]))
    
min_lat, min_long = min(min_lats)-0.25, min(min_longs)-0.25
max_lat, max_long = max(max_lats)+0.25, max(max_longs)+0.25


latitude = all_data[0].latitude[np.nonzero(all_data[0].latitude)]
longitude = all_data[0].longitude[np.nonzero(all_data[0].longitude)]

latitude = latitude[::100]
longitude = longitude[::100]

print("resample finish.")
#BBox = [-28.037, -20.786, 12.715, 17.545]
BBox = [min_long, max_long, min_lat, max_lat]
colors = ["gold", "orange", "darkolivegreen", "limegreen"]     

m = folium.Map(location=[max_lat, min_long], tiles="Stamen Terrain")
m.fit_bounds([[min_lat, min_long], [max_lat, max_long]])
points = []
for data in all_data:
    for i in range(len(latitude)):
        points.append((latitude[i], longitude[i]))
folium.PolyLine(points).add_to(m)    
j = 0
for group in current_groups:
    for segment in group:
        lat = segment.latitude[0]
        lon = segment.longitude[0]
        if len(segment.ax)<500:
            segment_radius = len(segment.ax)
        else:
            segment_radius = 500
            
        folium.Circle(
            radius=segment_radius**1.3,
            location=[lat, lon],
            color=colors[j],
            fill=True,
        ).add_to(m)
    j = j+1
    
print("points finish")
print("chivatillo")
m.save("mymap.html")