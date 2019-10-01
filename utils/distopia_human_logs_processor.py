from glob import glob
import sys
import os
sys.path.append(os.getcwd())
from data_types.distopia_data import DistopiaData


# filename convention is name_seed_task.json
# maybe create separate files for different users
# best would be to read json to distopia data and spit out npz's

name = "zhilong"

data_dir = ('/home/dev/data/distopia/team_logs')
logpaths = glob(os.path.join(data_dir,'{}_*.json'.format(name)))
data = DistopiaData()
data.set_params({'metric_names':['population','pvi','compactness'],'preprocessors':[]})
for logfile in logpaths:
    print(logfile)
    data.load_data(logfile,append=True,load_designs=False,load_metrics=True)
fname = os.path.join(data_dir,"{}_logs".format(name))
data.save_csv(fname,fname)