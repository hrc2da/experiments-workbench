class Experiment:
    def run(self,specs):
        '''run the experiment specified in specs
        Note that specs is (for now) a namespace object
        It should probably be a dict but I like dot operators...
        '''
        raise NotImplementedError

# add your experiment type here
import experiment_types.agent_experiment
import experiment_types.intent_filter_experiment
import experiment_types.classification_experiment
__all__ = ['agent_experiment','intent_filter_experiment','classification_experiment']