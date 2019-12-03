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
        self.batch_size = 32

    def set_params(self, specs_dict):
        self.num_metrics = specs_dict['num_metrics']
        if 'task' in specs_dict:
            task = specs_dict['task']
            if task == []:
                self.reward_weights = [1 for _ in range(self.nb_metrics)]
            else:
                assert len(task) == self.num_metrics
                self.reward_weights = task

        self.discount_rate = 0.9  # corresponds to gamma in the paper
        self.eps = 0.9 
        self.min_eps = 0.1
        self.num_episodes = 1
        self.episode_length = 100
        self.step_size = 10 # how many pixels to move in each step
        if 'num_episodes' in specs_dict:
            self.num_episodes = specs_dict['num_episodes']
        if 'episode_length' in specs_dict:
            self.episode_length = specs_dict['episode_length']
        if 'discount_rate' in specs_dict:
            self.discount_rate = specs_dict['discount_rate']
        if 'epsilon' in specs_dict:
            self.eps = specs_dict['epsilon']
        if 'min_eps' in specs_dict:
            self.min_eps = specs_dict['min_eps']
        if 'step_size' in specs_dict:
            self.step_size = specs_dict['step_size']

        self.n_steps = self.num_episodes * self.episode_length
        print("num episodes: " + str(self.num_episodes))
        print("episode length: " + str(self.episode_length))
        print("discount_rate: "  + str(self.discount_rate))

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
        self.model.add(Flatten(input_shape=(1,) + 40))
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
        """In our DQN these are the direction mappings:
            0: left
            1: right
            2: up
            3: down"""
        num = random.random()
        if self.eps<self.min_eps:
            self.eps = self.min_eps
        if num < self.eps:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state.reshape(-1, self.nb_actions))[0])
        return action

    def take_action(self, environment, action):
        """Take the action in the environment. 
            First unpack the action into block and move
            Returns the new state and reward"""
        block_num = action//4
        direction = action%4
        # try to make the move, if out of bounds, try again
        new_design = environment.make_move(block_num, direction)
        if new_design == -1 or environment.get_metrics(new_design) is None:
            print("ACTION OUT OF BOUNDS")
            reward = 0
            orig_design = environment.state
            new_state = self.get_state(environment, orig_design) # actually unchanged

        else: # valid move
            new_state = self.get_state(environment, new_design)
            old_metric = environment.get_metrics(environment.state)
            new_metric = environment.get_metrics(new_design)
            reward = environment.get_reward(new_metric-old_metric, self.reward_weights)
            environment.take_step(new_design)
        
        return new_state, reward

    def replay(self):
        """Gets a random batch from memory and replay"""
        batch = random.sample(self.memory, self.batch_size)
        for old_state, action, reward, next_state in batch:
            next_state_pred = self.model.predict(next_state.reshape(-1, 40)[0])
            old_state_pred = self.model.predict(old_state.reshape(-1, 40)[0])
            max_next_pred = np.max(next_state_pred)
            max_next_action = np.argmax(next_state_pred)
            target_q_value = reward + self.discount_rate * max_next_pred
            old_state_pred[max_next_action] = target_q_value
            self.model.fit(old_state.reshape(-1, 40)[0], old_state_pred, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state):
        """adds the experience into the dqn memory"""
        self.memory.append((state, action, reward, next_state))

    def fill_memory(self, environment, steps):
        """Run the game for defined number of steps with random actions"""
        for i in range(steps):
            rand_action = random.sample(self.action_space, 1)
            orig_state = self.get_state(environment, environment.state)
            new_state, reward = self.take_action(environment, rand_action)
            self.remember(orig_state, rand_action, reward, new_state)


    def train(self, environment):
        
        init_design = environment.reset()
        curr_state = self.get_state(environment, init_design)
        for i in range(self.num_episodes):
            # reset the block positions after every episode
            for j in range(self.episode_length):
                action = self.get_action(curr_state)
                next_state, reward = self.take_action(environment, action)
                # add this experience to memory
                self.remember(curr_state, action, reward, next_state)
                if j%10==0:
                    self.replay()  # do memory replay after every 10 steps
        # After training is done, save the model
        model_name = "trained_dqn_"+str(self.num_episodes)+"_"+str(self.episode_length)
        self.save_model(model_name)

    def run(self, environment):
        # FIrst ensure that there are enough experiences in memory to sample from
        self.fill_memory(environment, 100)
        environment.reset()
        self.train(environment)