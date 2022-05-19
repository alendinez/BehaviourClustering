import copy
import time
import numpy as np
import tsaug
import random

from tsaug.visualization import plot
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix

import models.data_manager as data_manager
import models.segment_manager as segment_manager
import models.network as Network

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_manager and segment_manager    
sigma = 6
w = 100
mode = "mean"
segment_manager = segment_manager(sigma, w, mode)
data_manager = data_manager()

output_path = "../Data/output/"

groups = np.load(output_path + "groups_raw.npy", allow_pickle = True)

### Cross-validation 1. Create train and test data for Reservoir Computing (80% train, 20% test).

temp_groups = copy.deepcopy(groups)
    
segments_train, segments_test = [], []

num_groups = len(groups)
max_group_length = int(max([len(group) for group in groups[1:len(groups)-1]]))

num_segments_train_pergroup = int(0.8*max_group_length)
num_segments_test_pergroup = max_group_length - num_segments_train_pergroup
num_segments_train = num_segments_train_pergroup*num_groups
num_segments_test = num_segments_test_pergroup*num_groups  # Number of segments from each group that we will use to train the network.
print("Number of train segments:", num_segments_train)
print("Number of test segments:", num_segments_test)

train_segments = []
test_segments = []
train_data_ax, train_data_ay, train_data_az, len_segments_train, labels_train = [], [], [], [], []
test_data_ax, test_data_ay, test_data_az, len_segments_test, labels_test = [], [], [], [], []          

### Perform data augmentation in order to have the same number of examples from each behavioral group.
for group in temp_groups:
    while len(group) < max_group_length:
        try:
            current_segment = copy.copy(random.choice(group))
            
            csa_ax = tsaug.AddNoise(scale=0.025).augment(current_segment.ax)
            csa_ay = tsaug.AddNoise(scale=0.025).augment(current_segment.ay)
            csa_az = tsaug.AddNoise(scale=0.025).augment(current_segment.az)
                
            current_segment.ax, current_segment.ay, current_segment.az = csa_ax, csa_ay, csa_az
            group.append(current_segment)
        except:
            continue
print("Data augmentation performed")

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
print("Segments set in order")            
   
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
print("Data is ready for training")

### Train and test Reservoir Computer Network
Network = Network.Network()
num_nodes = 300

input_probability = 0.5
reservoir_probability = 0.5
classifier = "log"

Network.T = sum(len_segments_train)  
Network.n_min = 1
Network.K = 3
Network.N = num_nodes

Network.setup_network(train_data, num_nodes, input_probability, reservoir_probability, num_groups, num_segments_train)
print("Training network...")
Network.train_network(num_groups, classifier, num_segments_train, len_segments_train, labels_train, num_nodes)

Network.mean_test_matrix = np.zeros([Network.N, num_segments_test])
Network.test_network(test_data, num_segments_test, len_segments_test, num_nodes, num_groups, sum(len_segments_test))

if classifier == 'log':
    print(f'Performance using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,labels_test.T)}')
    prediction = Network.regressor.predict(Network.mean_test_matrix.T)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

### Plot confusion matrix
print("Plotting confusion matrix...")
fig, ax = plt.subplots(figsize = (12,9))
disp = plot_confusion_matrix(Network.regressor, Network.mean_test_matrix.T, labels_test.T, normalize='true', ax=ax)
disp.ax_.set_title("Confusion matrix")
plt.show()

