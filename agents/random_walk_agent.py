from agents import Agent
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
import random
from matplotlib import pyplot as plt
import os

class RWAgent(Agent):
    def __init__(self):
        print('initializing random walk agent')
        self.reward_weights = []
        self.nb_actions = 32
        self.nb_metrics = 3
        self.action_space = np.arange(0, 32)
        self.total_pop = 0
        self.max_pvi = 246977.5

    def set_params(self, specs_dict):
        self.num_metrics = specs_dict['num_metrics']
        if 'task' in specs_dict:
            task = specs_dict['task']
            if task == []:
                self.reward_weights = [1 for _ in range(self.nb_metrics)]
            else:
                assert len(task) == self.num_metrics
                self.reward_weights = task

        self.num_episodes = 1
        self.episode_length = 100
        self.step_size = 10 # how many pixels to move in each step
        self.init_explore = 50
        if 'num_episodes' in specs_dict:
            self.num_episodes = specs_dict['num_episodes']
        if 'episode_length' in specs_dict:
            self.episode_length = specs_dict['episode_length']
        if 'step_size' in specs_dict:
            self.step_size = specs_dict['step_size']
        self.n_steps = self.num_episodes * self.episode_length
        print("num episodes: " + str(self.num_episodes))
        print("episode length: " + str(self.episode_length))
        
    def set_task(self, task):
        assert len(task) == self.num_metrics
        self.reward_weights = task

    def get_action(self, environment):
        """In our DQN these are the direction mappings:
            0: left
            1: right
            2: up
            3: down"""
        """Returns a LEGAL action (following epsilon greedy policy)"""
        done = False
        while done==False:
            action = random.randint(0, 31)
            block_num = action//4
            direction = action%4
            new_design = environment.make_move(block_num, direction)
            if new_design == -1 or environment.get_metrics(new_design) is None:
                continue
            else:
                reward = self.take_action(environment, action)
                done=True
        return action, reward

    def take_action(self, environment, action):
        """Take the action in the environment during evaluation; we assume its not an illegal state"""
        block_num = action//4
        direction = action%4
        # try to make the move, if out of bounds, try again
        new_design = environment.make_move(block_num, direction)
        assert new_design!=environment.state
        new_metric = environment.get_metrics(new_design)
        reward = environment.get_reward(new_metric, self.reward_weights)
        environment.take_step(new_design)

        return reward

    def evaluate_model(self, environment, num_steps, num_episodes, initial=None):

        print("Evaluating on seed 0")
        environment.seed(0)
        final_reward=[]
        for n in range(num_episodes):
            print(n)
            environment.reset(initial, max_blocks_per_district = 1)
            # init_design = environment.state
            rewards_log = []
            for i in range(num_steps):
                action, reward = self.get_action(environment)
                rewards_log.append(reward)
            final_reward.append(rewards_log[-1]  - rewards_log[0])
        mean = np.mean(final_reward)
        var = np.var(final_reward)
        print("MEAN OF EVAL REWARDS: ", mean)
        print("VARIANCE OF EVAL REWARDS: ", var)
        plt.plot(final_reward)
        plt.savefig(os.path.join(os.getcwd(),"{}.png".format("random_walk_agent")))


    def run(self, environment, status=None, initial=None):
        # FIrst ensure that there are enough experiences in memory to sample from
#        environment.reset(initial, max_blocks_per_district = 1)
        self.evaluate_model(environment, 100, 100, None)
        exit(0)
        # return train_design_log, train_metric_log, train_reward_log, self.num_episodes

