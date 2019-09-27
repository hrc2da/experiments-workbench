from experiment_types import Experiment

class AgentExperiment:
    def run(self, specs):
        specs['agent'] = specs['agent']()
        specs['environment'] = specs['environment']()
        specs['environment_params']['logfile'] = specs['logpath']
        specs['agent_params']['logfile'] = specs['logpath']
        specs['environment'].set_params(specs['environment_params'])
        specs['agent'].set_params(specs['agent_params'])
        return specs['agent'].run(specs['environment'],specs['n_steps'])
