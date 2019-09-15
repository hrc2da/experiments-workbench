from experiment_types import Experiment

class IntentFilterExperiment:
    def run(self, specs):
        # load trajectory data

        # split into train/test

        # train model on training data

        # test model on test data



        specs.environment_params['logfile'] = specs.logpath
        specs.agent_params['logfile'] = specs.logpath
        specs.environment.set_params(specs.environment_params)
        specs.runner.set_params(specs.agent_params)
        specs.runner.run(specs.environment,specs.n_steps)
