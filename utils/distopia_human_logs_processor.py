from glob import glob
import sys
import os
sys.path.append(os.getcwd())
from data_types.distopia_data import DistopiaData


# filename convention is name_seed_task.json
# maybe create separate files for different users
# best would be to read json to distopia data and spit out npz's

name = "10_06_23_27_41"

data_dir = ('/home/zhilong/Documents/HRC/team_logs/')

is_human = 0;
logpaths = glob(os.path.join(data_dir,'{}.json'.format(name)))
data = DistopiaData()
data.set_params({'metric_names':['population','pvi','compactness','projected_votes'],'preprocessors':[]})
for logfile in logpaths:
    print(logfile)
    if is_human:
        data.load_data(logfile,append=True,load_designs=False,load_metrics=True)
    else:
        data.load_agent_data(logfile,append=True,load_designs=False,load_metrics=True)
fname = os.path.join(data_dir,"{}_logs".format(name))
data.save_csv(fname,fname)