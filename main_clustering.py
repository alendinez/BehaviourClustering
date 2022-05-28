import os
import copy
import time
import random
import numpy as np
import multiprocessing
from tslearn.clustering import KShape

import models.data_manager as data_manager
import models.segment_manager as segment_manager
import models.KShapeVariableLength as KShapeVariableLength

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    start_time = time.time()

    sigma = 0.3
    w = 150
    mode = "std"
    segment_manager = segment_manager(sigma, w, mode)
    data_manager = data_manager()
    
    path = "../Data/output/"

    ### Load all data
    all_data = np.load(path + "all_data.npy", allow_pickle = True)
    print("Data loaded")

    ### Load previously created acceleration segments
    all_segments = data_manager.load_all_segments_linux(path, sigma, w)
    for data in all_data:
        for segment in all_segments:
            if segment.filename == data.filename:
                segment.setup_acceleration(data)
    print("Acceleration data set")

    ### Segments filtering
    all_segments.sort(key=lambda x: len(x.ax), reverse=True)
    l = len(all_segments)
    segments_in = [x.id for x in all_segments[int(l*0.05):]]
    segments_in.sort()
    assert segments_in[0] == min(segments_in)
    print("Segments filtered")
    
    ### Load correlation data
    maxcorr = np.load(path + f"maxcorr_{sigma}_{w}.npy")
    maxcorr = maxcorr[np.ix_(segments_in, segments_in)]
    print("Max correlation matrix loaded:", maxcorr.shape)

    ### Dummy matrix
    X = np.zeros((maxcorr.shape[0], 100, 3))

    inertias = []
    # Create the model and fit it
    for i in range(21,31):
        ksvl = KShapeVariableLength(n_clusters=i, max_iter=100, n_init=20)
        ksvl.fit(X, cross_dists=maxcorr)

        ksvl.to_hdf5(path + f'ksvl_{i}_clusters.hdf5')
        inertias.append(ksvl.inertia_)
        print(f'{i} clusters inertia:', ksvl.inertia_)

    print(inertias)
    
    finish_time = time.time()
    total_time = finish_time - start_time
    print("Computing time:",total_time, "seconds.")