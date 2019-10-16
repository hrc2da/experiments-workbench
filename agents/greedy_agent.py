from agents import Agent
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
from copy import deepcopy
import time
#import tqdm
import itertools

class GreedyAgent(Agent):

    def __init__(self):
        print('initializing greedy agent')
        self.reward_weights = []

    def seed(self,seed):
        np.random.seed(seed)

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
        if logger is None:
            metric_log = []
            mappend = metric_log.append
            design_log = []
            dappend = design_log.append
            reward_log = []
            rappend = reward_log.append
        while i < n_steps:
            i += 1
            if i % 100 == 0:
                last_reward = float("-inf")
                environment.reset(initial)
            count = 0
            best_reward_this_step = []
            best_metrics_this_step = []
            best_neighborhood_this_step = []
            best_reward_val_this_step = float("-inf")
            for j in range(n_tries_per_step):
                # clearing metrics and rewards to prevent case where
                # we continue on empty neighborhood and end loop without clearing reward
                # I think this is causing the index error
                metrics = []
                rewards = []
                samples += 1
                neighborhood = environment.get_sampled_neighborhood(4,2)
                if len(neighborhood) < 1:
                    continue
                else:
                    metrics = [environment.get_metrics(n, exc_logger) for n in neighborhood]
                    count += len(metrics)
                    rewards = [environment.get_reward(m, self.reward_weights) for m in metrics]
                    best_idx = np.argmax(rewards)
                    # if there are no legal and evaluatable moves, ignore this try
                    if rewards[best_idx] == float("-inf"):
                        no_valids += 1
                    # if on the other hand there is a move that beats the last step
                    # the first step, this is any legal move
                    elif rewards[best_idx] > last_reward:
                        break
                    # otherwise, record this sample in case we can't find a better one
                    # skip if it's worse than the best seen so far
                    elif len(best_reward_this_step) == 0 or rewards[best_idx] > best_reward_val_this_step:
                        best_reward_this_step = rewards[:]
                        best_metrics_this_step = deepcopy(metrics)
                        best_reward_val_this_step = rewards[best_idx]
                        best_neighborhood_this_step = deepcopy(neighborhood)
            assert len(rewards) == len(neighborhood)
            # if we ended and didn't find something better, take the last best legal thing we saw
            # however, if there's no legal states then just reset
            #if rewards[best_idx] <= last_reward or rewards[best_idx] == float("-inf"):
            if len(rewards) == 0 or rewards[best_idx] == float("-inf"):
                if len(best_reward_this_step) > 0:
                    rewards = best_reward_this_step[:]
                    metrics = deepcopy(best_metrics_this_step)
                    neighborhood = deepcopy(best_neighborhood_this_step)
                    best_idx = np.argmax(rewards)
                else:
                    last_reward = float("-inf")
                # if rewards[best_idx] == float("-inf"):
                #     print("No valid moves! Resetting!")
                # else:
                #     print("No better move! Resetting!")
                # what should I do here? this means there's nowhere to go that's legal
                    i -= 1 # not sure if this is right, but get the step back. will guarantee n_steps
                # alternatives, restart and add an empty row, or just skip this step
                    resets += 1
                    environment.reset(initial)
                    continue
            if np.random.rand() < eps:
                randoms += 1
                # mask out the legal options
                legal_mask = np.array([1 if r > float("-inf") else 0 for r in rewards], dtype=np.float32)
                # convert to probability
                legal_mask /= np.sum(legal_mask)
                best_idx = np.random.choice(np.arange(len(rewards)), p=legal_mask)
            if eps > eps_min:
                eps *= eps_decay
            if eps < eps_min:
                eps = eps_min
            last_reward = rewards[best_idx]
            # TODO: need to update occupied when changing state
            # chosen_neighbor = neighborhood[best_idx]
            # for district in chosen_neighbor:
            #     for block in chosen_neighbor[district]:
            #         if block not in self.state[district]:
            #             self.occupied.add(block)
            # for district in self.state:
            #     for block in self.state[district]:
            #         if block not in chosen_neighbor[district]:
            #             self.occupied.remove(block)
            environment.take_step(neighborhood[best_idx])
            environment.occupied = set(itertools.chain(*environment.state.values()))
            if status is not None:
                status.put('next')
            if logger is not None:
                logger.write(str([time.perf_counter(), count, list(metrics[best_idx]), environment.state]) + '\n')
            else:
                mappend(metrics[best_idx])
                dappend(environment.state)
                rappend(rewards[best_idx])
        # normalize metrics
        norm_metrics = []
        for m in metric_log:
            norm_metrics.append(environment.standardize_metrics(m))
        if logger is not None:
            return "n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights
        else:
            print("n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights)
            return design_log, norm_metrics, reward_log
