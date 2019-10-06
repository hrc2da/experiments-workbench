'''
Creates an npy file for x using metrics, y using labels
Only keeps data for labels that are in the subset we are using
'''

import sys
import os
sys.path.append(os.getcwd())
from data_types.distopia_data import DistopiaData

input_path = "/home/dev/data/distopia/243_raw.pkl"
target_path = "/home/dev/data/distopia/100_for_243_design_targets11"
label_path = "/home/dev/data/distopia/100_for_243_design_labels11"

data = DistopiaData()
data.set_params({"preprocessors":[], "n_workers":7, "slice_lims": [1100,1200]})
data.load_data(input_path)