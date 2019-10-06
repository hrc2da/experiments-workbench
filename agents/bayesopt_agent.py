from agents import Agent
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
from copy import deepcopy
import time
#import tqdm
import itertools
from skopt import forest_minimize
from skopt.space import Real

# TODO: make sure the task is passed in in spec (write set_params I guess)
# TODO: write code to rip params relevant to the optimizer and ** them in the run
# TODO: for the optimizer fn, just call utils.import_class(skopt.gp_minimize) for example

class BayesOptAgent(Agent):
    def setup_dimensions(self,x_min,x_max,y_min,y_max,n_blocks):
        dimensions = []
        for i in range(n_blocks):
            dimensions.append(Real(x_min,x_max))
            dimensions.append(Real(y_min,y_max))
        return dimensions

    def reward_fn(self,state):
        stripped_state = self.environment.fixed2dict(state)
        metrics = self.environment.get_metrics(stripped_state)
        return self.environment.get_reward(metrics,self.task)

    def run(self,environment,n_steps,params=None):
        '''
            runs environment.step n_steps times
            selects the action to take on each step
        '''    
        import pdb; pdb.set_trace()
        max_blocks = 5
        self.environment = environment
        init_state = self.environment.reset()
        x0 = self.environment.dict2fixed(init_state,max_blocks)
        y0 = self.environment.get_reward(init_state)
        dimensions = self.setup_dimensions(self.environment.x_min,
                                            self.environment.x_max,
                                            self.environment.y_min,
                                            self.environment.y_max,max_blocks)
        results = self.optimizer_fn(self.reward_fn,dimensions,num_calls=n_steps,
                                        x0=x0, y0=y0, **self.params)
        import pdb; pdb.set_trace()
        
        