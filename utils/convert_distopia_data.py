import sys
import os
sys.path.append(os.getcwd())
from data_types.distopia_data import DistopiaData

input_path = "/home/dev/data/distopia/243_raw.pkl"
target_path = "/home/dev/data/distopia/100_for_243_design_targets11"
label_path = "/home/dev/data/distopia/100_for_243_design_labels11"

data = DistopiaData()
data.set_params({"preprocessors":["truncate_design_dict","design_dict2mat_labelled"], "n_workers":7, "slice_lims": [1100,1200]})
data.load_data(input_path)

data.save_npy(target_path+".npz",label_path+".npz")
data.save_csv(target_path+".csv",label_path+".csv")