from agents import *
from environments import *
from experiment_types import *
from data_types import *
from models import *
from utils import import_class
import yaml
import os
import sys

run_spec_file = 'run_spec.yaml'
with open(run_spec_file, 'r') as input_stream:
    run_spec = yaml.safe_load(input_stream)


def import_params(param_file):

    with open(param_file, 'r') as input_stream:
        param_dict = yaml.safe_load(input_stream)
        if "experiment_description" not in param_dict.keys() or param_dict["experiment_description"] == '':
            raise(ValueError("No description specified for experiment '{}'. Please add 'experiment_description' to the experiment spec file.".format(param_file)))
        # check the experiment type
        # note that this only checks at the module, and not the class level.
        if "experiment_type" not in param_dict.keys() or param_dict["experiment_type"].split('.')[1] not in sys.modules['experiment_types'].__all__:
            raise(ValueError("Experiment type '{}' is not defined or unrecognized. Please define an experient type in the experiment_types module (see __init__.py for details)".format(param_dict["experiment_type"])))
        param_dict = parse_param_types(param_dict,0,1)
    return param_dict

def parse_param_types(param_dict, parse_level=0, max_parse_level=0, soft=True):
    for param, param_val in param_dict.items():
            if param not in run_spec:
                if soft == False:
                    raise(ValueError("Param '{}' is not in the run spec. Please add it to run_spec.yaml.".format(param)))
            else:
                param_spec = run_spec[param]
                if "type" in param_spec:
                    if param_spec["type"] == "Object" or param_spec["type"] == "runnable":
                        param_dict[param] = eval(param_val)
                    elif param_spec["type"] == "import":
                        param_dict[param] = import_class(param_val)
                    elif type(param_val) == dict:
                        # recursively parse dicts if not at max specified depth (this is so we don't have to define library specific specs)
                        if parse_level < max_parse_level:
                            param_dict[param] = parse_param_types(param_val,parse_level+1,max_parse_level)
                        # let it be otherwise (we return below)
                    else:
                        param_dict[param] = eval(param_spec["type"])(param_val)
                else:
                    if soft == False:
                        raise(ValueError("No type specified for param '{}'. Please add it to run_spec.yaml".format(param)))
    return param_dict

def setup_log_dir(spec_path):
    if os.name == 'nt':
        fpath = spec_path.split('\\')[:-1]
        fname = spec_path.split('\\')[-1].split('.')[0] # get the filename stripping out .spec.yaml
    else:
        fpath = spec_path.split('/')[:-1]
        fname = spec_path.split('/')[-1].split('.')[0] # get the filename stripping out .spec.yaml

    log_path = os.path.join(*fpath,'logs',fname)
    if os.path.exists(log_path):
        raise("Experiment log for '{}' already exists! Please choose a unique name or delete the old log directory.".format(log_path))
    os.mkdir(log_path)
    os.rename(spec_path,os.path.join(log_path,fname + '.spec.yaml'))
    return log_path
