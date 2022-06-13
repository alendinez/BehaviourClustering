import sys
sys.path.insert(0,"..") ## Set path to main directory

import os
import time
import numpy as np
import datetime
import copy
import random
import pickle
from scipy import signal
from functools import partial

import models.data as dt
import models.data_manager as data_manager
import models.segment_manager as segment_manager
import models.segment as sgmnt
import models.KShapeVariableLength as KShapeVariableLength

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()


output_path = "../../Data/output/"

groups = np.load(output_path + "groups_raw_12_3.npy", allow_pickle = True)

with open(output_path + 'groups_raw_12_3.pkl', 'wb') as outp:
    pickle.dump(groups, outp, pickle.HIGHEST_PROTOCOL)


finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")