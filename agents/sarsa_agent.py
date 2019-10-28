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
        """Return the coordinates of all the 8 blocks in a list of (rol, col)"""
        state_coords = []
        for i in range(np.size(q_table, 0)):
            block_x, block_y = state[i][0]
            row = np.size(q_table,2) - 1 - (block_y-boundries[2])
            col = block_x-boundries[0]

            state_coords.append((row, col))
        return state_coords

    def next_action(self, q_table, boundries, environment, eps, eps_min, eps_decay):
        num_blocks = np.size(q_table, 0)
        num_actions = np.size(q_table, 1)
        cur_state = environment.state
        state_coords  = self.get_state_coords(q_table, boundries, cur_state)
        # Now actions will be a |blocks|x|actions| array instead
        actions_array = np.random.rand(num_blocks, num_actions)
        for j, c in enumerate(state_coords):
            row, col = c
            actions = [q_table[j][i][row][col] for i in range(num_actions)]
            actions_array[j, :] = actions
        # best_action is a tuple of (block#, direction)
        done=False
        while done==False:
            if np.random.rand() < eps:
                best_action = (np.random.randint(0, num_blocks), np.random.randint(0, num_actions))
            else:
                best_action = np.unravel_index(np.argmax(actions_array), actions_array.shape)
            proposed_design = environment.make_move(best_action[0], best_action[1])
            if proposed_design == -1:
                print("ACTION OUT OF BOUNDS...TRYING AGAIN")
                actions_array[best_action[0],best_action[1]] = -100000 #Guarantees it won't be picked again
            elif environment.get_metrics(proposed_design) is None:
                old_eps = eps
                eps = 1
            else:
                done=True
        if eps == 1:
            eps = old_eps
        if eps > eps_min:
            eps *= eps_decay
        if eps < eps_min:
            eps = eps_min
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
        # 4D Q table to account for all 8 blocks
        q_table = np.random.rand(8, 4, game_boundries[3] - game_boundries[2] + 1 , game_boundries[1] - game_boundries[0]+ 1)
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

            # removed random restart after 100 steps.
            old_state = environment.state
            stepped_design = environment.make_move(best_action[0], best_action[1])
            metric = environment.get_metrics(stepped_design)
            reward = environment.get_reward(metric, self.reward_weights)
            # go back to the q table to update the reward of taking this step
            environment.take_step(stepped_design)
            next_action = self.next_action(q_table, game_boundries, environment, eps, eps_min, eps_decay)
            curr_state_coords = self.get_state_coords(q_table, game_boundries, old_state)
            next_state_coords = self.get_state_coords(q_table, game_boundries, stepped_design)

            state_row, state_col = curr_state_coords[best_action[0]] #best_action[0] = block, best_action[1] = move
            next_row, next_col = next_state_coords[next_action[0]]

            q_table[best_action[0], best_action[1], state_row, state_col] = \
                learning_coeff * \
                (reward + discount_coeff*q_table[next_action[0], next_action[1], next_row, next_col] - q_table[best_action[0], best_action[1], state_row, state_col])
            # TODO: Don't know if the above update is correct given the 4d table

            environment.occupied = set(itertools.chain(*environment.state.values()))
            best_action = next_action
            if status is not None:
                status.put('next')
            if logger is not None:
                logger.write(str([time.perf_counter(), i, list(metric), environment.state]) + '\n')
            else:
                mappend(metric)
                dappend(environment.state)
                rappend(reward)

        # normalize metrics
        norm_metrics = []
        # for m in metric_log:
        #     norm_metrics.append(environment.standardize_metrics(m))
        if logger is not None:
            return "n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights
        else:
            print("n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights)
            return design_log, metric_log, reward_log
