import copy
import time
import numpy as np
import segment_manager
import data_manager
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

start_time = time.time()

### Initialize data_manager and segment_manager    
sigma = 4
w = 50
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)
data_manager = data_manager.data_manager()

data_path = "../Data/CSV/"
output_path = "../Data/output/"

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

print("Starting...")
for filename in filenames:
    # Load data and filter acceleration signals with a butterworth filter
    data = data_manager.load_data(filename, data_path)
    data.filter_accelerations(4, 0.4)
    all_data.append(data)
    print("Data loaded: "+filename)


groups_raw = np.load(output_path + "groups_raw.npy", allow_pickle = True)
lag_ax = np.load(output_path + "lag_ax.npy")

### Check the number of groups with more than 100 elements
plus100 = 0
total_segments = 0
segments_in_100 = 0
for group in groups_raw:
    if len(group) >= 100:
        plus100 = plus100+1
        segments_in_100 = segments_in_100+len(group)
    total_segments = total_segments+len(group)
        
percentage_in_100 = (segments_in_100/total_segments)*100
percentage_out_100 = 100 - (segments_in_100/total_segments)*100
print("Number of groups with more than 100 elements: "+str(plus100))
print("Percentage of segments in these groups: "+str(percentage_in_100))
print("Percentage of segments out of these groups: "+str(percentage_out_100))

### Save N most common behaviors
N = 10
groups = segment_manager.save_most_common_behaviors(groups_raw, N)
print(N, "most common behaviours selected")

### Align segments from the same group
groups = segment_manager.align_segments(groups, lag_ax)

### Set up acceleration for the aligned segments
''' En teoria viene ya con el set
for data in all_data:
    for group in groups:
        for segment in group:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
'''

### Find some group metrics
segment_manager.find_group_metrics(groups, all_data)

### Find average behavior for each group in the three axis and plot it
avrg_group_ax, avrg_group_ay, avrg_group_az, avrg_group_pressure = segment_manager.find_average_behavior(groups, mode="nanmean")

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
    
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")
