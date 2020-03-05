from agents import Agent
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
from tensorflow import keras
from keras.models import Sequential, model_from_yaml, Model
from keras.layers import Dense, Activation, Flatten, Input, Multiply
from keras.optimizers import Adam
import random
from matplotlib import pyplot as plt
import os
from utils import ringbuffer

class DQNAgentMask(Agent):
    def __init__(self):
        print('initializing DQN agent')
        self.reward_weights = []
        self.nb_actions = 32
        self.nb_metrics = 3
        self.optimizer = Adam(lr=0.001)
        self.action_space = np.arange(0, 32)
        self.batch_size = 32
        self.build_model()
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

        self.discount_rate = 0.9  # corresponds to gamma in the paper
        self.eps = 0.95
        self.min_eps = 0.1
        self.decay_rate = 0.999
        self.num_episodes = 1
        self.episode_length = 100
        self.step_size = 10 # how many pixels to move in each step
        self.memory_size = 10000 #Size of replay buffer
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
        if 'buffer_size' in specs_dict:
            self.dequeue_size = specs_dict['buffer_size']
        self.memory = ringbuffer.RingBuffer(self.memory_size)
        self.n_steps = self.num_episodes * self.episode_length
        print("num episodes: " + str(self.num_episodes))
        print("episode length: " + str(self.episode_length))
        print("discount_rate: "  + str(self.discount_rate))
        print("buffer_size: "  + str(self.dequeue_size))
    def set_task(self, task):
        assert len(task) == self.num_metrics
        self.reward_weights = task

    def seed(self,seed):
        np.random.seed(seed)

    def get_state(self, environment, design):
        """returns the current positions of the 8 blocks and the metrics
            The output data structure is 8x4 array: 1 row for each block,
            each row is of the form [x_pos, y_pos, pop_metric, pvi_metric, compact_metric]"""

        ret_state = np.zeros((8, 5))
        district_metrics = environment.get_district_metrics(design)
        if self.total_pop==0: # so that we don't recompute everytime
            for i in range(8):
                self.total_pop+=district_metrics[i][0]
        for i in range(8):
            block_x, block_y = design[i][0]
            # normalize metrics and pixels
            block_x = (block_x-environment.x_min)/(environment.x_max-environment.x_min)
            block_y = (block_y-environment.y_min)/(environment.y_max-environment.y_min)
            ret_state[i][0] = block_x
            ret_state[i][1] = block_y
            ret_state[i][2] = district_metrics[i][0]/self.total_pop
            ret_state[i][3] = district_metrics[i][1]/self.max_pvi
            ret_state[i][4] = district_metrics[i][2]
        return ret_state

    def build_model(self):

        state_input = Input(shape=(40,))
        actions_input = Input(shape=(self.nb_actions,))

        x1 = Dense(64, activation='relu')(state_input)
        x2 = Dense(64, activation='relu')(x1)
        x3 = Dense(self.nb_actions, activation='linear')(x2)
        out = Multiply()([x3, actions_input])

        self.model = Model(inputs = [state_input, actions_input], outputs = out)
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def save_model(self, filename):
        with open(filename+".yaml", 'w') as yaml_file:
            yaml_file.write(self.model.to_yaml())
        self.model.save_weights(filename+".h5")

    def get_action(self, state, environment):
        """In our DQN these are the direction mappings:
            0: left
            1: right
            2: up
            3: down"""
        """Returns a LEGAL action (following epsilon greedy policy)"""
        done = False
        all_actions_mask = np.ones((32,))
        predict_q = self.model.predict([state.reshape(1,40), all_actions_mask.reshape(1,32)])[0]
        while done==False:
            num = random.random()
            if self.eps<self.min_eps:
                self.eps = self.min_eps
            if num < self.eps:
                best_action = np.random.choice(self.action_space)
            else:
                best_action = np.argmax(predict_q)
            block_num = best_action//4
            direction = best_action%4
            new_design = environment.make_move(block_num, direction)
            if new_design == -1:
                predict_q[best_action] = np.NINF #Guarantees won't be picked again
            else:
                new_metric = environment.get_metrics(new_design)
                if new_metric is None:
                    predict_q[best_action] = np.NINF #Guarantees won't be picked again
                else:
                    new_state, reward = self.take_action(environment, new_design, new_metric)
                    done=True
        self.eps *= self.decay_rate
        return best_action, new_state, new_metric, reward

    def take_action(self,environment, new_design, new_metric):
        """Take the action in the environment during evaluation; we assume its not an illegal state"""
        # try to make the move, if out of bounds, try again
        # print("TAKING ACTION! Moving block {}, in {} direction".format(block_num, direction))
        new_state = self.get_state(environment, new_design)
        reward = environment.get_reward(new_metric, self.reward_weights)
        environment.take_step(new_design)

        return new_state, reward


    # def replay(self):
    #     """Gets a random batch from memory and replay"""
    #     batch = random.sample(self.memory, self.batch_size)
    #     for old_state, action, reward, next_state in batch:
    #         next_state_pred = self.model.predict(next_state.reshape(1, 40))[0]
    #         # print(next_state_pred)
    #         old_state_pred = self.model.predict(old_state.reshape(1, 40))[0]
    #         # print(old_state_pred)
    #         max_next_pred = np.max(next_state_pred)
    #         max_next_action = np.argmax(next_state_pred)
    #         target_q_value = reward + self.discount_rate * max_next_pred
    #         old_state_pred[action] = target_q_value
    #         # as an optimization, try not un-shaping and re_shaping old_state_pred
    #         self.model.fit(old_state.reshape(1, 40), old_state_pred.reshape(1, 32), epochs=1, verbose=0)

    def replay(self):
        #Sample from memory, isolate into different columns
        batch = self.memory.sample(self.batch_size)
        old_states = [y[0] for y in batch]
        actions = [y[1] for y in batch]
        one_hot_actions = [[0]* self.nb_actions for y in batch]
        rewards = [y[2] for y in batch]
        next_states = [y[3] for y in batch]
        for i in range(len(batch)):
            old_states[i] =  old_states[i].reshape(40,)
            next_states[i] = next_states[i].reshape(40,)
            one_hot_actions[i][actions[i]] = 1
        next_states_q_vals = self.model.predict([next_states,np.ones((self.batch_size, self.nb_actions))])
        max_q_next_states = np.max(next_states_q_vals,axis=1)
        real_q_vals = rewards + max_q_next_states*self.discount_rate

        self.model.fit([old_states,one_hot_actions], one_hot_actions * real_q_vals[:,None], use_multiprocessing=True, verbose=0)



    def remember(self, state, action, reward, next_state):
        """adds the experience into the dqn memory"""
        self.memory.append((state, action, reward, next_state))


    def evaluate_q(self, rand_states, environment):
        all_actions_mask = np.ones((32,))
        max_vals=[]
        for rand_state in rand_states:
            actual_state = self.get_state(environment, rand_state)
            predict_q = self.model.predict([actual_state.reshape(1,40), all_actions_mask.reshape(1,32)])[0]
            max_vals.append(np.max(predict_q))
        return (sum(max_vals)/len(max_vals))

    def train(self, environment, status=None, initial=None):

        train_reward_log = []
        train_metric_log = []
        train_design_log = []
        random_states = []
        q_progress = []
        # init_design={}
        # init_design[0] = [(540,570)]
        # init_design[1] = [(740,770)]
        # init_design[2] = [(770,190)]
        # init_design[3] = [(310,460)]
        # init_design[4] = [(800,220)]
        # init_design[5] = [(680,750)]
        # init_design[6] = [(490,560)]
        # init_design[7] = [(470,350)]
        for i in range(50):
            environment.reset(None, max_blocks_per_district = 1)
            rand_design = environment.state
            random_states.append(rand_design)
        q_progress.append(self.evaluate_q(random_states, environment))

        # environment.reset(None, max_blocks_per_district = 1)
        # init_design = environment.state
        old_reward = 0
        for i in range(self.num_episodes):
            # reset the block positions after every episode
            environment.reset(None, max_blocks_per_district = 1)
            init_design = environment.state
            curr_state = self.get_state(environment, init_design)
            print(init_design)
            for j in range(self.episode_length):
                action, next_state, metric, reward = self.get_action(curr_state, environment)
                train_reward_log.append(reward)
                train_design_log.append(environment.state)
                train_metric_log.append(metric)
                # add this experience to memory
                self.remember(curr_state, action, reward-old_reward, next_state)
                old_reward = reward
                if j%5==0 and len(self.memory) >= self.batch_size:
                    self.replay()  # do memory replay after every 5 steps
                if status is not None:
                    status.put('next')
            q_progress.append(self.evaluate_q(random_states, environment))

        # After training is done, save the model
        model_name = "trained_dqn_mask_"+str(self.num_episodes)+"_"+str(self.episode_length)
        self.save_model(model_name)
        plt.plot(q_progress)
        plt.savefig(os.path.join(os.getcwd(),"{}.png".format("agent_dqn_mask_q_progress")))
        return train_design_log, train_metric_log, train_reward_log

    def evaluate_model(self, environment, num_steps, num_episodes, initial=None):
        """Use the currently trained model to play distopia from a random
            starting state for num_steps steps, plot the metrics"""

        print("Evaluating on seed 43")
        environment.seed(43)

        filename = "trained_dqn_mask_"+str(1000)+"_"+str(100)
        with open(filename + '.yaml', 'r') as f:
            self.model = model_from_yaml(f.read())
        self.model.load_weights(filename +'.h5')

        final_reward=[]
        for _ in range(num_episodes):
            environment.reset(initial, max_blocks_per_district = 1)
            init_design = environment.state
            curr_state = self.get_state(environment, init_design)
            print(curr_state)
            rewards_log = []
            for i in range(num_steps):
                self.eps = 0 #Ensures no random choice is made
                _,next_state, _, reward = self.get_action(curr_state,environment)
                rewards_log.append(reward)
                curr_state = next_state
            final_reward.append(sum(rewards_log)/len(rewards_log))
        plt.ylim(-3.0,3.0)
        plt.plot(final_reward)
        print("AVG REWARD MASK: ")
        print(sum(final_reward)/len(final_reward))
        plt.savefig(os.path.join(os.getcwd(),"{}.png".format("agent_dqn_mask_eval_reward")))


    def run(self, environment, status=None, initial=None):
        # FIrst ensure that there are enough experiences in memory to sample from
#        environment.reset(initial, max_blocks_per_district = 1)
        train_design_log, train_metric_log, train_reward_log = self.train(environment, status, initial)
        print("Training done..Evaluating: ")
#        self.evaluate_model(environment, 100, 100, None)
        return train_design_log, train_metric_log, train_reward_log, self.num_episodes
