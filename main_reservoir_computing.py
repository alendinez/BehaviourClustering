import copy
import time
import numpy as np
import tsaug
import random
import os

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
sigma = 0.3
w = 150
mode = "std"
segment_manager = segment_manager(sigma, w, mode)
data_manager = data_manager()

output_path = "../Data/output/"

all_data = np.load(output_path + "all_data.npy", allow_pickle = True)
print("Data loaded")

### Load previously created acceleration segments
all_segments = data_manager.load_all_segments_linux(output_path, sigma, w)
for data in all_data:
    for segment in all_segments:
        if segment.filename == data.filename:
            segment.setup_acceleration(data)
print("Acceleration data set")

### Segments filtering
segments_copy = all_segments.copy()
segments_copy.sort(key=lambda x: len(x.ax), reverse=True)
l = len(segments_copy)
segments_out = [x.id for x in segments_copy[:int(l*0.05)]]
segments_out.sort(reverse=True)
for idx in segments_out:
    all_segments.pop(idx)
print("Segments filtered")

i = 0
for segment in all_segments:
    segment.id = i
    i = i+1
print("Segments reindexed")

groups_idx = np.load(output_path + "groups_raw_16_4.npy", allow_pickle = True)
groups =  []
for i in range(len(groups_idx)):
    segments = []
    for idx in groups_idx[i]:
        sgmnt = all_segments[idx]
        sgmnt.group_label = i
        segments.append(sgmnt)
    groups.append(segments)

'''
groups = np.load(output_path + "groups_raw_2.npy", allow_pickle = True)
'''

### Cross-validation 1. Create train and test data for Reservoir Computing (80% train, 20% test).

augmented_groups = copy.deepcopy(groups)
    
segments_train, segments_test = [], []

num_groups = len(groups)
max_group_length = int(max([len(group) for group in groups]))

num_segments_train_pergroup = int(0.8*max_group_length)
num_segments_test_pergroup = max_group_length - num_segments_train_pergroup
num_segments_train = num_segments_train_pergroup*num_groups
num_segments_test = num_segments_test_pergroup*num_groups  # Number of segments from each group that we will use to train the network.
print("Number of train segments:", num_segments_train)
print("Number of test segments:", num_segments_test)

train_data_ax, train_data_ay, train_data_az, len_segments_train, labels_train = [], [], [], [], []
test_data_ax, test_data_ay, test_data_az, len_segments_test, labels_test = [], [], [], [], []          

### Perform data augmentation in order to have the same number of examples from each behavioral group.
for group in augmented_groups:
    while len(group) < max_group_length:
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
            current_segment.id = None
            group.append(current_segment)
        except:
            continue
print("Data augmentation performed")

temp_groups = copy.copy(augmented_groups)

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
        group.remove(current_segment)
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

del train_data_ax
del train_data_ay
del train_data_az
del test_data_ax
del test_data_ay
del test_data_az
del groups
del augmented_groups
del temp_groups

### Train and test Reservoir Computer Network
Network = Network.Network()
num_nodes = 200

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
    prediction_test = Network.regressor.predict(Network.mean_test_matrix.T)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

### Plot confusion matrix
print("Plotting confusion matrix...")
fig, ax = plt.subplots(figsize = (12,9))
disp = plot_confusion_matrix(Network.regressor, Network.mean_test_matrix.T, labels_test.T, normalize='true', ax=ax)
disp.ax_.set_title("Confusion matrix")
plt.show()




Network.mean_test_matrix = np.zeros([Network.N, num_segments_train])
Network.test_network(train_data, num_segments_train, len_segments_train, num_nodes, num_groups, sum(len_segments_train))

if classifier == 'log':
    print(f'Performance using {classifier} over train data: {Network.regressor.score(Network.mean_test_matrix.T,labels_train.T)}')
    prediction_train = Network.regressor.predict(Network.mean_test_matrix.T)

### Plot confusion matrix
print("Plotting confusion matrix...")
fig, ax = plt.subplots(figsize = (12,9))
disp = plot_confusion_matrix(Network.regressor, Network.mean_test_matrix.T, labels_train.T, normalize='true', ax=ax)
disp.ax_.set_title("Confusion matrix of train data")
plt.show()


del Network


pipeline_output = [segments_train, segments_test, labels_train, labels_test, prediction_train, prediction_test]
np.save(os.path.join(output_path, 'pipeline_output.npy'), pipeline_output)