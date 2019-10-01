from glob import glob
import sys
import os
sys.path.append(os.getcwd())
from data_types.distopia_data import DistopiaData
from pathlib import Path
from matplotlib import pyplot as plt
from environments.distopia_environment import DistopiaEnvironment

name = "*"

data_dir = Path.home().joinpath('data','distopia','team_logs')
input_glob = glob(os.path.join(str(data_dir),'{}_*logs.csv'.format(name)))





def plot_trajectory(trajectory,title=""):
    vars = zip(*trajectory)
    for v in vars:
        plt.plot(v)
    plt.title(title)
    plt.show()



data = DistopiaData()
data.set_params({'metric_names':['population','pvi','compactness'],'preprocessors':[]})
for f in input_glob:
    fn = Path(f)
    fn_root = fn.with_suffix('')
    data_fn = str(fn.with_suffix(".csv")) # TODO: gross, clean this up
    task_fn = str(fn_root)+"_labels.csv"
    data.load_data(data_fn,labels_path=task_fn,append=True)

print(data.x.shape)
print(data.y.shape)
task_dict = data.get_task_dict()
for k,v in task_dict.items():
    print(len(v))
    for run in v:
        plot_trajectory(run,k)
n_keys = len(task_dict.values())
print(n_keys)


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