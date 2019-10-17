from pathlib import Path
import sys
import os
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
from data_types.distopia_data import DistopiaData
import numpy as np
import pickle as pkl


# generate a mean/std vector for all metrics from a set of agent json log files
if (len(sys.argv) != 2):
    print("USAGE: python distopia_generate_standardization_file.py <data path>")
    exit(0)
data_dir = sys.argv[1]
is_human = 0
logpaths = sorted(Path(data_dir).glob('**/*.json'))
print("Found {} logfiles to process.".format(len(logpaths)))


data = DistopiaData()
cur_metrics = ['population', 'pvi', 'compactness']
data.set_params({'metric_names': cur_metrics,'preprocessors':[]})

for log in logpaths:
    data.load_agent_data(log,append=True,load_designs=False,load_metrics=True)

mean = np.mean(data.x,axis=0)
std = np.std(data.x,axis=0)

with open(os.path.join(data_dir,"standardization_params.pkl"), 'wb') as outfile:
    pkl.dump((mean,std),outfile)



# for i in range(len(logpaths)):
#     logfile = logpaths[i]
#     print(logfile)
#     new_user=False
#     if os.name == "nt":
#         cur_file = logfile.split('\\')[-1]
#     else:
#         cur_file = logfile.split('/')[-1]
#     file_attr = cur_file.split('_')
#     if file_attr[0] == "agent":
#         is_human=0
#     else:
#         is_human=1
#     if i < len(logpaths)-1:
#         if os.name == "nt":
#             next_file = logpaths[i+1].split('\\')[-1]
#         else:
#             next_file = logpaths[i+1].split('/')[-1]
#         next_file_attr = next_file.split('_')
#     if i==len(logpaths)-1 or file_attr[0] != next_file_attr[0]:
#         new_user=True
#     if is_human:
#         data.load_data(logfile,append=True,load_designs=False,load_metrics=True, norm_file = norm_dir)
#     else:
#         data.load_agent_data(logfile,append=True,load_designs=False,load_metrics=True)
#     if new_user == True:
#         fname = os.path.join(data_dir,"{}_logs".format(file_attr[0]))
#         print(fname)
#         data.save_csv(fname,fname)
#         del data
#         data= DistopiaData()
#         data.set_params({'metric_names': cur_metrics,'preprocessors':[]})
