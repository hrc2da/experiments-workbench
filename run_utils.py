from agents import *
from environments import *
import yaml
from argparse import Namespace
import os

run_spec_file = 'run_spec.yaml'
with open(run_spec_file, 'r') as input_stream:
    run_spec = yaml.safe_load(input_stream)


def import_params(param_file):

    with open(param_file, 'r') as input_stream:
        param_dict = yaml.safe_load(input_stream)
        if "experiment_description" not in param_dict.keys() or param_dict["experiment_description"] == '':
            raise(ValueError("No description specified for experiment '{}'. Please add 'experiment_description' to the experiment spec file.".format(param_file)))
        for param, param_val in param_dict.items():
            if param not in run_spec:
                raise(ValueError("Param '{}' is not in the run spec. Please add it to run_spec.yaml.".format(param)))
            else:
                param_spec = run_spec[param]
                if "type" in param_spec:
                    if param_spec["type"] == "Object" or param_spec["type"] == "runnable":
                        param_dict[param] = eval(param_val)
                    elif type(param_val) == dict:
                        continue # don't recursively parse dicts for now. leave to whatever receives/uses it
                    else:
                        param_dict[param] = eval(param_spec["type"])(param_val)
                else:
                    raise(ValueError("No type specified for param '{}'. Please add it to run_spec.yaml".format(param)))
    return Namespace(**param_dict)

def setup_log_dir(spec_path):
    fpath = spec_path.split('/')[:-1]
    fname = spec_path.split('/')[-1].split('.')[0] # get the filename stripping out .spec.yaml
    log_path = os.path.join(*fpath,'logs',fname)
    if os.path.exists(log_path):
        raise("Experiment log for '{}' already exists! Please choose a unique name or delete the old log directory.".format(log_path))
    os.mkdir(log_path)
    os.rename(spec_path,os.path.join(log_path,fname + '.spec.yaml'))
    return log_path
                