from glob import glob
import sys
import os
#sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
sys.path.append(os.path.normpath(os.getcwd()))
from data_types.distopia_data import DistopiaData
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from environments.distopia_environment import DistopiaEnvironment

# name = "10_09_23_29_18"
def plot_trajectory(trajectory, title="", prefix="", names=[],outdir=None):
    labels = names
    vars = list(zip(*trajectory))
    for i in range(len(vars)-len(names)):
        labels.append('')
    assert len(labels) == len(vars)
    print(len(vars))
    plt.clf()
    for i,v in enumerate(vars):
        plt.plot(v,label=labels[i])
    plt.title(title)
    plt.legend()
    if outdir:
        modified_title = prefix + "_" + title
        plt.savefig(os.path.join(outdir,"{}.png".format(modified_title)))
    else:
        plt.show()
# def trajectories2task_dict(x,y):
def avg_trajectories(trajectories,truncate=True):
    lengths = [len(traj) for traj in trajectories]
    min_len = min(lengths)
    avg_traj = []
    for i in range(min_len):
        step_vals = [t[i] for t in trajectories]
        avg_traj.append(np.mean(step_vals))
    if truncate == True:
        return avg_traj

def plot_metrics(data_dir):
    metrics = ['population', 'pvi', 'compactness']
    input_glob = sorted(glob(os.path.join(str(data_dir),'*logs.csv')))
    output_dir = os.path.join(str(data_dir),"images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = DistopiaData()
    data.set_params({'metric_names':metrics,'preprocessors':[]})
    new_user = False
    for i in range(len(input_glob)):
        f= input_glob[i]
        if os.name == "nt":
            cur_file = f.split('\\')[-1]
        else:
            cur_file = f.split('/')[-1]
        cur_file_attr = cur_file.split('_')
        if cur_file_attr[-1] == 'logs.csv':
            log_index=-1
        if cur_file_attr[-1] == 'labels.csv':
            log_index=-2
        title_prefix = "_".join(cur_file_attr[0:log_index])

        if i < len(input_glob)-1:
            if os.name == "nt":
                next_file = input_glob[i+1].split('\\')[-1]
            else:
                next_file = input_glob[i+1].split('/')[-1]
            next_file_attr = next_file.split('_')
            if next_file_attr[-1] == 'logs.csv':
                next_log_index=-1
            if next_file_attr[-1] == 'labels.csv':
                next_log_index=-2
            next_title_prefix = "_".join(next_file_attr[0:next_log_index])

        if i==len(input_glob)-1 or title_prefix != next_title_prefix:
            new_user=True
        fn = Path(f)
        fn_root = fn.with_suffix('')
        data_fn = str(fn.with_suffix(".csv")) # TODO: gross, clean this up
        task_fn = str(fn_root)+"_labels.csv"
        data.load_data(data_fn,labels_path=task_fn,append=True)
        if new_user == True:
            print(data.x.shape)
            print(data.y.shape)
            task_dict = data.get_task_dict()
            for k,v in task_dict.items():
                print(len(v)),
                for vrun in v:
                    plot_trajectory(trajectory = vrun,title = k, names = metrics, outdir = str(output_dir), prefix =title_prefix)
            n_keys = len(task_dict.values())
            print(n_keys)
            del data
            data = DistopiaData()
            data.set_params({'metric_names':metrics,'preprocessors':[]})
            new_user = False
