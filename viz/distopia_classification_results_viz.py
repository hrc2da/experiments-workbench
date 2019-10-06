import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from data_types.distopia_data import DistopiaData

logdir = "experiments/logs"
experiment_name ="distopia_cnn_designs_human_test_1"

resultsdir = os.path.join(logdir,experiment_name)


test_labels = np.load(os.path.join(resultsdir,"test_labels.npy"))
test_predictions = np.load(os.path.join(resultsdir,"test_predictions.npy"))


unique_labels = []
label_strings = []

for label in test_labels:
    label_string = DistopiaData.task_arr2str(label)
    if label_string not in label_strings:
        unique_labels.append(label)
        label_strings.append(label_string)

for label in unique_labels:
    indices = np.where((test_labels == label).all(axis=1))[0]
    mean_pred = np.mean(test_predictions[indices],axis=0)
    plt.bar(range(len(label)),mean_pred)
    plt.title(str(label))
    plt.show()