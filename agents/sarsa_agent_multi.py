from agents import Agent
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
from copy import deepcopy
import time
#import tqdm
import itertools
import pickle

class SARSAAgent(Agent):

    def __init__(self):
        print('initializing SARSA agent multi-update')
        self.reward_weights = []
    def set_params(self, specs_dict):
        self.num_metrics = specs_dict['num_metrics']
        if 'task' in specs_dict:
            task = specs_dict['task']
            if task == []:
                self.reward_weights = [1 for _ in range(num_metrics)]
            else:
                assert len(task) == self.num_metrics
                self.reward_weights = task

        self.learning_coeff = 0.2
        self.discount_coeff = 0.9
        self.num_episodes = 1
        self.episode_length = 100
        self.subsample_scale = 1
        if 'num_episodes' in specs_dict:
            self.num_episodes = specs_dict['num_episodes']
        if 'episode_length' in specs_dict:
            self.episode_length = specs_dict['episode_length']
        if 'learning_coeffecient' in specs_dict:
            self.learning_coeff = specs_dict['learning_coeffecient']
        if 'discount_coeffecient' in specs_dict:
            self.discount_coeff = specs_dict['discount_coeffecient']
        if 'subsample_scale' in specs_dict:
            self.subsample_scale = specs_dict['subsample_scale']

        self.n_steps = self.num_episodes * self.episode_length
        print("Reward weights: ")
        print(self.reward_weights)
        print("num episodes: " + str(self.num_episodes))
        print("episode length: " + str(self.episode_length))
        print("learning_coeff: "  + str(self.learning_coeff))
        print("discount_coeff: " + str(self.discount_coeff))
        print("subsample scale agent: " + str(self.subsample_scale))
    def set_task(self, task):
        assert len(task) == self.num_metrics
        self.reward_weights = task

    def seed(self,seed):
        np.random.seed(seed)
        
    def get_state_coords(self, q_table, boundries, state):
        """Return the coordinates of all the 8 blocks in a list of (rol, col) in the q-table
            With subsampling, this becomes more involved, need to first subsample the map
            coordinates and then map to the q-table."""
        state_coords = []
        for i in range(np.size(q_table, 0)):
            block_x, block_y = state[i][0]
            row = np.where(self.possible_y==block_y)[0][0]
            col = np.where(self.possible_x==block_x)[0][0]
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
            if j != 0:
                actions_array[j,:] = np.NINF

        # best_action is a tuple of (block#, direction)
        done=False
        while done==False:
            use_eps=False
            if np.random.rand() < eps:
                best_action = (0, np.random.randint(0, num_actions))
                use_eps = True
#                best_action = (np.random.randint(0, num_blocks), np.random.randint(0, num_actions))
            else:
                all_max = np.argwhere(actions_array == np.amax(actions_array))
                best_ind = np.random.randint(0, np.size(all_max,0))
#                best_action = np.unravel_index(np.argmax(actions_array), actions_array.shape)
                best_action= all_max[best_ind]
            proposed_design = environment.make_move(best_action[0], best_action[1], use_eps)
            if proposed_design == -1 or environment.get_metrics(proposed_design) is None:
#                print("ACTION OUT OF BOUNDS...TRYING AGAIN")
                actions_array[best_action[0],best_action[1]] = np.NINF #Guarantees it won't be picked again
            else:
                done=True
        if eps > eps_min:
            eps *= eps_decay
        if eps < eps_min:
            eps = eps_min
        return best_action, eps, use_eps

    def run(self, environment, logger=None, exc_logger=None, status=None, initial=None, eps=0.8, eps_decay=0.9,
            eps_min=0.1, n_tries_per_step = 10):
        '''runs for n_steps and returns traces of designs and metrics
        '''
        if logger is None and hasattr(self,'logger') and self.logger is not None:
            logger = self.logger
        subsample_scale = self.subsample_scale

        j = 0
        last_reward = float("-inf")
        no_valids = 0
        samples = 0
        resets = 0
        randoms = 0
        # initialize q-table. Hold rewards in a |actions|x|states| in a numpy array
        # encode actions as follows: {0: block0 up, 1: block0 down, 2: block0 left, 3: block0 right,
        # 4: block1 up ..., 31: block7 right} --> for now only block0 moves
        game_boundries=environment.get_boundaries()
        self.possible_x = np.arange(game_boundries[2], game_boundries[3]+1, subsample_scale)
        self.possible_y = np.arange(game_boundries[0], game_boundries[1]+1, subsample_scale)

        #If we are moving only one block, there are only (x_max-x_min) * (y_max - y_min) states
        # 4D Q table to account for all 8 blocks
        subsampled_x = len(self.possible_x)
        subsampled_y = len(self.possible_y)

#        q_table = np.random.rand(8, 4, subsampled_x, subsampled_y)
        q_table = np.full((8,4,subsampled_x, subsampled_y), 5.0)
        print('Q TABLE SIZE: ', q_table.shape)
        if logger is None:
            metric_log = []
            mappend = metric_log.append
            design_log = []
            dappend = design_log.append
            reward_log = []
            rappend = reward_log.append

        # initial = {}
        # initial[0] = [(640, 900)]
        # initial[1] = [(680, 480)]
        # initial[2] = [(280, 300)]
        # initial[3] = [(580, 900)]
        # initial[4] = [(520, 700)]
        # initial[5] = [(180, 640)]
        # initial[6] = [(360, 580)]
        # initial[7] = [(520, 420)]
        # print(initial)
        environment.reset(initial, max_blocks_per_district = 1)
        initial_state = deepcopy(environment.state)
        print(initial_state)
        while j < self.num_episodes:
            environment.reset(initial, max_blocks_per_district = 1)
#            environment.state = deepcopy(initial_state)
            j +=1
            best_action, eps, use_eps = self.next_action(q_table, game_boundries, environment, eps, eps_min, eps_decay)
            i=0
            while i < self.episode_length:
                # Logic for the sarsa agent:
                # at each step, get all the neighbors and compute the rewards and metrics, put into q table
                i += 1
                old_state = environment.state
#                print(best_action)
                stepped_design = environment.make_move(best_action[0], best_action[1], use_eps)
                metric = environment.get_metrics(stepped_design)
                reward = environment.get_reward(metric, self.reward_weights)

                # go back to the q table to update the reward of taking this step
                environment.take_step(stepped_design)
                next_action, eps, use_eps  = self.next_action(q_table, game_boundries, environment, eps, eps_min, eps_decay)
                curr_state_coords = self.get_state_coords(q_table, game_boundries, old_state)
                next_state_coords = self.get_state_coords(q_table, game_boundries, stepped_design)

                state_row, state_col = curr_state_coords[best_action[0]] #best_action[0] = block, best_action[1] = move
                next_row, next_col = next_state_coords[next_action[0]]
#                print("OLD Q:  " + str(q_table[best_action[0], best_action[1], state_row, state_col]))
                q_table[best_action[0], best_action[1], state_row, state_col] += \
                    self.learning_coeff * \
                    (reward + self.discount_coeff*q_table[next_action[0], next_action[1], next_row, next_col] - q_table[best_action[0], best_action[1], state_row, state_col])
                other_actions = [x for x in range(np.size(q_table,1)) if x != best_action[1]]
                for x in other_actions:
                    new_state_row = state_row
                    new_state_col = state_col
                    if x == 0: #move right
                        new_state_col = new_state_col - 1
                    if x == 1: #move left
                        new_state_col = new_state_col + 1
                    if x == 2: #move up
                        new_state_row  = new_state_row - 1
                    if x == 3: #move down
                        new_state_row = new_state_row + 1
                    if new_state_row >= 0 and new_state_row < np.size(q_table,2):
                        if new_state_col >= 0 and new_state_col < np.size(q_table,3):
                            q_table[best_action[0], x, new_state_row, new_state_col] += \
                                    self.learning_coeff * \
                                    (reward + self.discount_coeff*q_table[next_action[0], next_action[1], next_row, next_col] - q_table[best_action[0], x, new_state_row, new_state_col])

#                print("NEW Q: " + str(q_table[best_action[0], best_action[1], state_row, state_col]))

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
        with open("qtable_50.pkl", "wb") as f:
            pickle.dump(q_table, f)
        if logger is not None:
            return "n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(self.n_steps, samples, resets, no_valids, randoms), self.reward_weights
        else:
            print("n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(self.n_steps, samples, resets, no_valids, randoms), self.reward_weights)
            return design_log, metric_log, reward_log, self.num_episodes
