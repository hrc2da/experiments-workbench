import numpy as np
from glob import glob

combine_target_fpaths = glob("/home/dev/data/distopia/*targets*.npz.npy")
combine_label_fpaths = glob("/home/dev/data/distopia/*labels*.npz.npy")

combined_target_fpath = "/home/dev/data/distopia/100_for_243_design_targets_combined4.npy"
combined_label_fpath = "/home/dev/data/distopia/100_for_243_design_labels_combined4.npy"

print(combine_target_fpaths)
print(combine_label_fpaths)

targets = None
for f in combine_target_fpaths:
    data = np.load(f)
    if targets is None:
        targets = data
    else:
        targets = np.concatenate((targets,data))

labels = None
for f in combine_label_fpaths:
    data = np.load(f)
    if labels is None:
        labels = data
    else:
        labels = np.concatenate((labels,data))

np.save(combined_target_fpath,targets)
np.save(combined_label_fpath,labels)