import glob
import yaml
from run_utils import setup_log_dir, import_params
import os

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
    specs = import_params(fpath)
    logpath = setup_log_dir(fpath)

    specs['logpath'] = logpath

    print("log path is {}".format(logpath))

    # print("Running experiment {}: \n\t{}".format(fpath.split('/')[-1],specs.experiment_description))
    # print("Specs for experiment are: \n{}\n{}".format(line_sep,yaml.dump(specs)))
    # specs.experiment_type.run(specs)


    print("Running experiment {}: \n\t{}".format(fpath.split('\\')[-1],specs['experiment_description']))
    specs['experiment_type']().run(specs)
