import glob
import yaml
from run_utils import setup_log_dir, import_params
import os
import json
import datetime

if os.name == 'nt':
    experiment_spec_fpaths = glob.glob("experiments/*.spec.yaml")
else:
    experiment_spec_fpaths = glob.glob("./experiments/*.spec.yaml")

line_sep = "="*80
print(line_sep)
print("Running experiments in ")
print("Found experiment spec files:{}".format([fpath.split('/')[-1] for fpath in experiment_spec_fpaths]))
print(line_sep)

for fpath in experiment_spec_fpaths:
    log_data = {}
    specs = import_params(fpath)
    logpath = setup_log_dir(fpath)
    specs.logpath = logpath
    print("Running experiment {}: \n\t{}".format(fpath.split('\\')[-1],specs.experiment_description))
    log_data['experiment_info'] = specs.experiment_description
    log_data['start_time'] = datetime.datetime.now()
    log_data['metric_names'] = specs.environment_params['metrics']
    log_data['task'] = specs.agent_params['task']
    log_data['run_log'] = []
    curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    file_name = curr_time + '.json'
    metric_names = specs.environment_params['metrics']
    design, metric = specs.experiment_type.run(specs)
    for ind, m in enumerate(metric):
        this_step = {}
        this_step['step_no'] = ind
        this_step['metrics'] = m
        this_step['design'] = design[ind]
        log_data['run_log'].append(this_step)
    log_data['end_time'] = datetime.datetime.now()
    with open(os.path.join(logpath, file_name), 'w') as outfile:
        json.dump(log_data, outfile, default=str)