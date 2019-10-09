class Agent:

    def set_params(self,params):
        '''
        Takes a dict of params and sets each as an instance variable
        '''
        for param, param_val in params.items():
            setattr(self, param, param_val)
    def run(self,environment,n_steps,params=None):
        '''
            runs environment.step n_steps times
            selects the action to take on each step
        '''
        raise NotImplementedError

# add any agents you write to this list and __all__
import agents.greedy_agent
# import agents.bayesopt_agent
__all__ = ['greedy_agent']#,'bayesopt_agent']
