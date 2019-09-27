from experiment_types import Experiment
import datetime
import os
import json


class AgentExperiment:
    def run(self, specs):
        specs.environment_params['logfile'] = specs.logpath
        specs.agent_params['logfile'] = specs.logpath
        specs.environment.set_params(specs.environment_params)
        specs.agent.set_params(specs.agent_params)
        log_data = {}  # init log data
        log_data['experiment_info'] = specs.experiment_description
        log_data['start_time'] = datetime.datetime.now()
        log_data['metric_names'] = specs.environment_params['metrics']
        log_data['task'] = specs.agent_params['task']
        log_data['run_log'] = []
        curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        file_name = curr_time + '.json'
        metric_names = specs.environment_params['metrics']
        design, metric, reward = specs.agent.run(specs.environment,specs.n_steps)
        for ind, m in enumerate(metric):
            this_step = {}
            this_step['step_no'] = ind
            this_step['metrics'] = m
            this_step['design'] = design[ind]
            this_step['reward'] = reward[ind]
            log_data['run_log'].append(this_step)
        log_data['end_time'] = datetime.datetime.now()
        logpath = specs.logpath
        with open(os.path.join(logpath, file_name), 'w') as outfile:
            json.dump(log_data, outfile, default=str)

