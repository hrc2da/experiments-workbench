import time
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
from matplotlib import pyplot as plt
import os
from utils import ringbuffer
import pickle

class DDQNAgent(Agent):
    def __init__(self):
        print('initializing Double DQN agent')
        self.reward_weights = []
        self.nb_actions = 32
        self.nb_metrics = 3
        self.optimizer = Adam(lr=0.001)
        self.action_space = np.arange(0, 32)
        self.batch_size = 32
        self.tau = 0.9 # soft update parameter for the target network
        self.build_model()
        self.build_target_model()
        self.total_pop = 0
        self.max_pvi = 246977.5
        self.skip_steps = 1
        self.replay_steps = 5
        self.evaluate_q_steps = 1
        self.target_update_steps = 100


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
        # self.model = Sequential()
        # self.model.add(Dense(64, input_dim=40))
        # self.model.add(Activation('relu'))
        # self.model.add(Dense(64))
        # self.model.add(Activation('relu'))
        # self.model.add(Dense(self.nb_actions))
        # self.model.add(Activation('linear'))
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        filename = "trained_ddqn_"+str(9600)+"_"+str(100)
        with open(filename + '.yaml', 'r') as f:
            self.model = model_from_yaml(f.read())
        self.model.load_weights(filename +'.h5')
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def build_target_model(self):
        # self.target_model = Sequential()
        # self.target_model.add(Dense(64, input_dim=40))
        # self.target_model.add(Activation('relu'))
        # self.target_model.add(Dense(64))
        # self.target_model.add(Activation('relu'))
        # self.target_model.add(Dense(self.nb_actions))
        # self.target_model.add(Activation('linear'))
        # self.target_model.compile(loss='mse', optimizer=self.optimizer)
        filename = "trained_ddqn_"+str(9600)+"_"+str(100)
        with open(filename + '.yaml', 'r') as f:
            self.target_model = model_from_yaml(f.read())
        self.target_model.load_weights(filename +'.h5')
        self.target_model.compile(loss='mse', optimizer=self.optimizer)

    def save_model(self, filename):
        with open(filename+".yaml", 'w') as yaml_file:
            yaml_file.write(self.model.to_yaml())
        self.model.save_weights(filename+".h5")

    def save_target_model(self, filename):
        with open(filename+".yaml", 'w') as yaml_file:
            yaml_file.write(self.target_model.to_yaml())
        self.target_model.save_weights(filename+".h5")

    def save_buffer(self, filename):
        with open(filename+".pkl", "wb") as pkl_file:
            pickle.dump(self.memory, pkl_file)


    def get_action(self, state, environment, step_num, old_action):
        """In our DQN these are the direction mappings:
            0: left
            1: right
            2: up
            3: down
        Returns a LEGAL action (following epsilon greedy policy)"""
        nn_predict = False #Bool if neural network should predict move
        if step_num % self.skip_steps == 0 and old_action is not None:
            best_action = old_action
            block_num = best_action//4
            direction = best_action%4
            new_design = environment.make_move(block_num, direction)
            if new_design !=-1:
                new_metric = environment.get_metrics(new_design)
                if new_metric is not None:
                    new_state, reward = self.take_action(environment, new_design, new_metric)
                else:
                    nn_predict=True
            else:
                nn_predict=True
        else:
            nn_predict=True

        if nn_predict == True:
            done = False
            predict_q = self.model.predict(state.reshape(1, 40))[0]
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
                if new_design ==-1:
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

    def take_action(self, environment, new_design, new_metric):
        """Take the action in the environment during evaluation; we assume its not an illegal state"""
        # try to make the move, if out of bounds, try again
        # print("TAKING ACTION! Moving block {}, in {} direction".format(block_num, direction))
        # assert new_design!=environment.state
        new_state = self.get_state(environment, new_design)
        reward = environment.get_reward(new_metric, self.reward_weights)
        environment.take_step(new_design)

        return new_state, reward


    def replay(self):
        """Gets a random batch from memory and replay"""
        batch = self.memory.sample(self.batch_size)
        old_states = []
        old_state_preds = []
        for old_state, action, reward, next_state in batch:
            next_state_pred = self.model.predict(next_state.reshape(1, 40))[0]
            best_next_action = np.argmax(next_state_pred)
            target_pred = self.target_model.predict(next_state.reshape(1, 40))[0]
            target_q = target_pred[best_next_action]
            new_q_val = target_q*self.discount_rate+reward
            old_state_pred = self.model.predict(old_state.reshape(1, 40))[0]
            old_state_pred[action] = new_q_val
            old_states.append(old_state.reshape(1, 40))
            old_state_preds.append(old_state_pred.reshape(1, 32))
        old_states = np.array(old_states).reshape(self.batch_size, 40)
        old_state_preds = np.array(old_state_preds).reshape(self.batch_size, 32)
        self.model.fit(old_states, old_state_preds, batch_size=self.batch_size, epochs=1, use_multiprocessing=True, verbose=0)

    def update_target(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        # print("TAU IS: ", self.tau)
        # print("WEIGHTS ARE: ", model_weights)
        # print("shape of weights: ", model_weights.shape)
        model_updated = [self.tau*x for x in model_weights]
        target_updated = [(1-self.tau)*x for x in target_weights]
        target_weights = model_updated+target_updated
        self.target_model.set_weights(target_weights)


    def remember(self, state, action, reward, next_state):
        """adds the experience into the dqn memory"""
        self.memory.append((state, action, reward, next_state))


    def evaluate_q(self, rand_states, environment):
        max_qs=[]
        rewards = []
        for rand_state in rand_states:
            actual_state = self.get_state(environment, rand_state)
            done = False
            predict_q = self.model.predict(actual_state.reshape(1, 40))[0]
            while done==False:
                best_action = np.argmax(predict_q)
                block_num = best_action//4
                direction = best_action%4
                new_design = environment.make_move(block_num, direction)
                if new_design ==-1:
                    predict_q[best_action] = np.NINF #Guarantees won't be picked again
                else:
                    new_metric = environment.get_metrics(new_design)
                    if new_metric is None:
                        predict_q[best_action] = np.NINF #Guarantees won't be picked again
                    else:
                        reward = environment.get_reward(new_metric, self.reward_weights)
                        max_qs.append(predict_q[best_action])
                        rewards.append(reward)
                        done=True
        num_samples = len(max_qs)
        return sum(rewards)/num_samples, sum(max_qs)/num_samples


    def train(self, environment, status=None, initial=None):
        train_reward_log = []
        train_metric_log = []
        train_design_log = []
        random_states = []
        q_progress = []
        reward_progress = []

        for i in range(50):
            environment.reset(None, max_blocks_per_district = 1)
            rand_design = environment.state
            random_states.append(rand_design)
        eval_rewards, eval_qs = self.evaluate_q(random_states, environment)
        reward_progress.append(eval_rewards)
        q_progress.append(eval_qs)

        # environment.reset(None, max_blocks_per_district = 1)
        # init_design = environment.state
        for i in range(self.num_episodes):
            # reset the block positions after every episode
            environment.reset(None, max_blocks_per_district = 1)
            init_design = environment.state
            print(init_design)
            curr_state = self.get_state(environment, init_design)
            old_action = None #keeps track of last action
            old_reward = 0
            for j in range(self.episode_length):
                # action, next_state, metric, reward = self.get_action(curr_state,environment)
                # train_metric_log.append(metric)
                action, next_state, metric, reward = self.get_action(curr_state, environment, j, None)
                train_reward_log.append(reward)
                train_design_log.append(environment.state)
                train_metric_log.append(metric)
                # add this experience to memappendory
                self.remember(curr_state, action, reward-old_reward, next_state)
                old_reward = reward
                old_action = action
                if j%self.replay_steps==0 and len(self.memory) >= self.batch_size:
                    self.replay()  # do memory replay after every 10 steps
                if j%self.target_update_steps==0:
                    self.update_target() # update target network every 100 steps
                if status is not None:
                    status.put('next')
            if i % self.evaluate_q_steps == 0:
                eval_rewards, eval_qs = self.evaluate_q(random_states, environment)
                reward_progress.append(eval_rewards)
                q_progress.append(eval_qs)


        # After training is done, save the model
        model_name = "trained_ddqn_"+str(self.num_episodes*2)+"_"+str(self.episode_length)
        target_name = "trained_ddqn_"+str(self.num_episodes*2)+"_"+str(self.episode_length) + "_target"
        buffer_name = "trained_ddqn_"+str(self.num_episodes*2)+"_"+str(self.episode_length) + "_buffer"
        self.save_model(model_name)
        self.save_target_model(target_name)
        self.save_buffer(buffer_name)
        plt.plot(q_progress)
        plt.savefig(os.path.join(os.getcwd(),"{}.png".format("agent_ddqn_q_progress")))
        plt.close()
        plt.plot(reward_progress)
        plt.savefig(os.path.join(os.getcwd(),"{}.png".format("agent_ddqn_reward_progress")))
        return train_design_log, train_metric_log, train_reward_log

    def evaluate_model(self, environment, num_steps, num_episodes, initial=None):
        """Use the currently trained model to play distopia from a random
            starting state for num_steps steps, plot the metrics"""

        print("Evaluating on seed 43")
        environment.seed(43)
        self.skip_steps = 1
        filename = "trained_ddqn_"+str(9600*2)+"_"+str(100)
        with open(filename + '.yaml', 'r') as f:
            self.model = model_from_yaml(f.read())
        self.model.load_weights(filename +'.h5')
        with open(filename + '_target.yaml', 'r') as f:
            self.target_model = model_from_yaml(f.read())
        self.target_model.load_weights(filename +'_target.h5')

        final_reward=[]
        for e in range(num_episodes):
            print("EPISODE: ", e)
            environment.reset(initial, max_blocks_per_district = 1)
            init_design = environment.state
            curr_state = self.get_state(environment, init_design)
            # print(curr_state)
            # print(curr_state.reshape(1, 40))
            rewards_log = []
            for i in range(num_steps):
                print("STEP: ", i)
                _, next_state, _, reward = self.get_action(curr_state,environment,i,None)
                rewards_log.append(reward)
                curr_state = next_state
            final_reward.append(sum(rewards_log)/len(rewards_log))
        plt.ylim(-3.0,3.0)
        plt.plot(final_reward)
        print("AVG REWARD MASK: ")
        print(sum(final_reward)/len(final_reward))
        plt.savefig(os.path.join(os.getcwd(),"{}.png".format("agent_dqn_nomask_eval_reward")))


    def run(self, environment, status=None, initial=None):
        # FIrst ensure that there are enough experiences in memory to sample from
#        environment.reset(initial, max_blocks_per_district = 1)
        start = time.time()
        train_design_log, train_metric_log, train_reward_log = self.train(environment, status, initial)
        end = time.time()
        train_time = end - start
        print("Training done in " + str(train_time) + "seconds... Evaluating: ")#
#t         self.evaluate_model(environment, 5, 100, None)
        return train_design_log, train_metric_log, train_reward_log, self.num_episodes
