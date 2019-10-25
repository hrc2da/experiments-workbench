from agents import Agent
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
from copy import deepcopy
import time
#import tqdm
import itertools

class SARSAAgent(Agent):

    def __init__(self):
        print('initializing SARSA agent')
        self.reward_weights = []
        self.q_table = {}
    def set_params(self, specs_dict):
        self.num_metrics = specs_dict['num_metrics']
        if 'task' in specs_dict:
            task = specs_dict['task']
            if task == []:
                self.reward_weights = [1 for _ in range(num_metrics)]
            else:
                assert len(task) == num_metrics
                self.reward_weights = task

    def set_task(self, task):
        assert len(task) == self.num_metrics
        self.reward_weights = task


    def run(self, environment, n_steps, logger=None, exc_logger=None, status=None, initial=None, eps=0.8, eps_decay=0.9,
            eps_min=0.1, n_tries_per_step = 10):
        '''runs for n_steps and returns traces of designs and metrics
        '''
        if logger is None and hasattr(self,'logger') and self.logger is not None:
            logger = self.logger

        environment.reset(initial)
        i = 0
        last_reward = float("-inf")
        no_valids = 0
        samples = 0
        resets = 0
        randoms = 0
        # initialize q-table. Hold rewards in a |actions|x|states| in a numpy array
        # encode actions as follows: {0: block0 up, 1: block0 down, 2: block0 left, 3: block0 right, 
        # 4: block1 up ..., 31: block7 right}
        q_table = np.random.rand((32, 1))
        states = [environment.state] # use an array to keep track of states, built as we go

        if logger is None:
            metric_log = []
            mappend = metric_log.append
            design_log = []
            dappend = design_log.append
            reward_log = []
            rappend = reward_log.append
        while i < n_steps:
            # Logic for the sarsa agent:
            # at each step, get all the neighbors and compute the rewards and metrics, put into q table

            i += 1
            if i % 100 == 0:
                last_reward = float("-inf")
                environment.reset(initial)
            count = 0
            # take the max reward step from this current state
            best_action = np.argmax(q_table[:, -1])
            block_to_move = best_action//8
            direction = best_action%8
            stepped_design = environment.make_move(block_to_move, direction)
            metric = environment.get_metrics(stepped_design)
            reward = environment.get_reward(metric, self.reward_weights)
            # go back to the q table to update the reward of taking this step
            q_table[best_action, -1] = reward
            # TODO how to add a new row of rewards to the q table?
            
            states.append(stepped_design)  # append this new design

            environment.occupied = set(itertools.chain(*environment.state.values()))
            environment.take_step(stepped_design)

            
            if status is not None:
                status.put('next')
            if logger is not None:
                logger.write(str([time.perf_counter(), count, list(metric), environment.state]) + '\n')
            else:
                mappend(metric)
                dappend(environment.state)
                rappend(reward)


        # normalize metrics
        norm_metrics = []
        for m in metric_log:
            norm_metrics.append(environment.standardize_metrics(m))
        if logger is not None:
            return "n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights
        else:
            print("n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights)
            return design_log, norm_metrics, reward_log
