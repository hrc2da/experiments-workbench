from agents import Agent
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
from tensorflow import keras
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import random

class DQNAgent(Agent):
    def __init__(self):
        print('initializing DQN agent')
        self.reward_weights = []
        self.nb_actions = 32
        self.nb_metrics = 3
        self.optimizer = Adam(lr=0.001)
        self.memory = [] #TODO: we can also consider making this into a fixed-length queue
        self.action_space = np.arange(0, 32)

    def set_params(self, specs_dict):
        self.num_metrics = specs_dict['num_metrics']
        if 'task' in specs_dict:
            task = specs_dict['task']
            if task == []:
                self.reward_weights = [1 for _ in range(self.nb_metrics)]
            else:
                assert len(task) == self.num_metrics
                self.reward_weights = task

        self.learning_coeff = 0.2
        self.eps = 0.9  # TODO: change these parameters to more reasonable values
        self.min_eps = 0.1
        self.num_episodes = 1
        self.episode_length = 100
        if 'num_episodes' in specs_dict:
            self.num_episodes = specs_dict['num_episodes']
        if 'episode_length' in specs_dict:
            self.episode_length = specs_dict['episode_length']
        if 'learning_coeffecient' in specs_dict:
            self.learning_coeff = specs_dict['learning_coeffecient']
        if 'eps' in specs_dict:
            self.eps = specs_dict['eps']
        if 'min_eps' in specs_dict:
            self.min_eps = specs_dict['min_eps']

        self.n_steps = self.num_episodes * self.episode_length
        print("num episodes: " + str(self.num_episodes))
        print("episode length: " + str(self.episode_length))
        print("learning_coeff: "  + str(self.learning_coeff))
        print("discount_coeff: " + str(self.discount_coeff))

    def get_state(self, environment, design):
        """returns the current positions of the 8 blocks and the metrics
            The output data structure is 8x4 array: 1 row for each block, 
            each row is of the form [x_pos, y_pos, pop_metric, pvi_metric, compact_metric]"""
        ret_state = np.zeros((8, 4))
        district_metrics = environment.get_district_metrics(design)
        # TODO: need to check the format of the district_metrics. We want it to hold the 3 metrics for each district 
        for i in range(8):
            block_x, block_y = design[i][0]
            ret_state[i][0] = block_x
            ret_state[i][1] = block_y
            for j in range(3):
                ret_state[i][j] = district_metrics[i][j]
        return ret_state

    def build_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        # self.model.add(Dense(16))
        # self.model.add(Activation('relu'))
        self.model.add(Dense(self.nb_actions))
        self.model.add(Activation('linear'))
        self.model.compile(loss='mse', optimizer=self.optimizer)
    
    def save_model(self, filename):
        with open(filename+".yaml", 'w') as yaml_file:
            yaml_file.write(self.model.to_yaml())
        self.model.save_weights(filename+".h5")
    
    def get_action(self, state):
        num = random.random()
        if self.eps<self.min_eps:
            self.eps = self.min_eps
        if num < self.eps:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state.reshape(-1, self.nb_actions))[0])
        return action

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    