import time
import copy
import numpy as np

import models.data_manager

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

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
    initial_data = data_manager.load_data(filename, data_path)
    current_data = copy.deepcopy(initial_data)
    current_data.filter_accelerations(4, 0.4)
    all_data.append(current_data)
    print("Data loaded: " + filename)

### Extract the data
np.save(os.path.join(path, 'all_data.npy'), all_data)