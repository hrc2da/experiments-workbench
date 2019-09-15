import glob
import yaml
from run_utils import setup_log_dir, import_params

experiment_spec_fpaths = glob.glob("./experiments/*.spec.yaml")
line_sep = "="*80
print(line_sep)
print("Running experiments in ")
print("Found experiment spec files:{}".format([fpath.split('/')[-1] for fpath in experiment_spec_fpaths]))
print(line_sep)

for fpath in experiment_spec_fpaths:
    specs = import_params(fpath)
    logpath = setup_log_dir(fpath)
    print("Running experiment {}: \n\t{}".format(fpath.split('/')[-1],specs.experiment_description))
    print("Specs for experiment are: \n{}\n{}".format(line_sep,yaml.dump(specs)))
    if specs.experiment_type == 'agent':
        specs.environment_params['logfile'] = logpath
        specs.agent_params['logfile'] = logpath
        specs.environment.set_params(specs.environment_params)
        specs.runner.set_params(specs.agent_params)
        specs.runner.run(specs.environment,specs.n_steps)
    params = specs

