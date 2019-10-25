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
                assert len(task) == self.num_metrics
                self.reward_weights = task

    def set_task(self, task):
        assert len(task) == self.num_metrics
        self.reward_weights = task

    def get_state_coords(self, q_table, boundries, state):
        block_x, block_y = state[0][0]
        row = np.size(q_table,2) - (block_y-boundries[2])
        col = block_x-boundries[0]
        return row, col

    def next_action(self, q_table, boundries, environment, eps, eps_min, eps_decay):
        cur_state = environment.state
        row, col  = self.get_state_coords(q_table, boundries, cur_state)
        actions = [q_table[i][row][col] for i in range(np.size(q_table,0))]
        actions_array = np.array(actions)
        done=False
        while done==False:
            if np.random.rand() < eps:
                best_action = np.random.randint(0, len(actions_array))
            else:
                best_action = np.argmax(actions_array)
            if eps > eps_min:
                eps *= eps_decay
            if eps < eps_min:
                eps = eps_min
            if environment.make_move(0,best_action) == -1:
                print("DELETING ACTION...OUT OF BOUNDS")
                print(best_action)
                np.delete(actions_array, best_action)
            else:
                done=True
        return best_action

    def run(self, environment, n_steps, logger=None, exc_logger=None, status=None, initial=None, eps=0.8, eps_decay=0.9,
            eps_min=0.1, n_tries_per_step = 10, learning_coeff=0.2, discount_coeff = 0.9):
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
        # 4: block1 up ..., 31: block7 right} --> for now only block0 moves
        game_boundries=environment.get_boundaries()
        #If we are moving only one block, there are only (x_max-x_min) * (y_max - y_min) states
        q_table = np.random.rand(4, game_boundries[3] - game_boundries[2], game_boundries[1] - game_boundries[0])
#        states = [environment.state] # use an array to keep track of states, built as we go

        if logger is None:
            metric_log = []
            mappend = metric_log.append
            design_log = []
            dappend = design_log.append
            reward_log = []
            rappend = reward_log.append

        best_action = self.next_action(q_table, game_boundries, environment, eps, eps_min, eps_decay)
        while i < n_steps:
            # Logic for the sarsa agent:
            # at each step, get all the neighbors and compute the rewards and metrics, put into q table
            i += 1
            if i % 100 == 0:
                last_reward = float("-inf")
                environment.reset(initial)
            count = 0
            # take the max reward step from this current state
            #best_action = np.argmax(q_table[:, -1])
            #block_to_move = best_action//8
            #direction = best_action
            stepped_design = environment.make_move(0, best_action)
            metric = environment.get_metrics(stepped_design)
            reward = environment.get_reward(metric, self.reward_weights)
            # go back to the q table to update the reward of taking this step
            old_state = environment.state
            environment.take_step(stepped_design)
            next_action = self.next_action(q_table, game_boundries, environment, eps, eps_min, eps_decay)
            state_row, state_col = self.get_state_coords(q_table, game_boundries, old_state)
            next_row, next_col = self.get_state_coords(q_table, game_boundries, stepped_design)
            q_table[best_action, state_row, state_col] =  learning_coeff * (reward - discount_coeff*q_table[next_action, next_row, next_col] - q_table[best_action, state_row, state_col])
            # TODO how to add a new row of rewards to the q table?

    #        states.append(stepped_design)  # append this new design

            environment.occupied = set(itertools.chain(*environment.state.values()))
            best_action = next_action
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
