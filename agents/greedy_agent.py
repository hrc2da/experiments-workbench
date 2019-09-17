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
    # scalar_value is mean over districts
    # scalar_std is standard deviation between districts
    # scalar_maximum is max over districts
    # s is state metric object (for this metric)
    # d is list of district objects (for all metrics)
    metric_extractors = {

        #overall normalization plan: run one-hots in either direction to get rough bounds
        # then z-normalize and trim on edges
        #'population' : lambda s,d : s.scalar_std,
        # standard deviation of each district's total populations (-1)
        # normalization: [0, single_district std]
        'population' : lambda s,d : np.std([dm.metrics['population'].scalar_value for dm in d]),
        # mean of district margin of victories (-1)
        # normalization: [0,1]
        'pvi' : lambda s,d : s.scalar_maximum,
        # minimum compactness among districts (maximize the minimum compactness, penalize non-compactness) (+1)
        # normalization: [0,1]
        'compactness' : lambda s,d : np.min([dm.metrics['compactness'].scalar_value for dm in d]),
        # mean ratio of democrats over all voters in each district (could go either way)
        # normalization: [0,1]
        'projected_votes' : lambda s,d : np.mean([dm.metrics['projected_votes'].scalar_value/dm.metrics['projected_votes'].scalar_maximum for dm in d]),
        # std of ratio of nonminority to minority over districts
        # normalization: [0, ]
        'race' : lambda s,d : np.std([dm.metrics['race'].scalar_value/dm.metrics['race'].scalar_maximum for dm in d]),
        # scalar value is std of counties within each district. we take a max (-1) to minimize variance within district (communities of interest)
        'income' : lambda s,d : np.max([dm.metrics['income'].scalar_value for dm in d]),
        #'education' : lambda s,d : s.scalar_std,

        # maximum sized district (-1) to minimize difficulty of access
        # normalization [0,size of wisconsin]
        'area' : lambda s,d: s.scalar_maximum
    }

    def __init__(self, x_lim=(100, 900), y_lim=(100, 900),
                    step_size=5, step_min=50, step_max=100,
                    task=[],pop_mean=None,pop_std=None):
        print('initializing')
        self.x_min, self.x_max = x_lim
        self.y_min, self.y_max = y_lim
        self.step = step_size
        self.step_min = step_min
        self.step_max = step_max
        self.pop_mean = pop_mean
        self.pop_std = pop_std
        self.occupied = set()
        self.coord_generator = self.gencoordinates(self.x_min, self.x_max, self.y_min, self.y_max)
        self.evaluator = VoronoiAgent()
        self.evaluator.load_data()


    def set_params(self, specs_dict):
        metrics = specs_dict['metrics']
        if metrics == []:
            self.set_metrics(self.evaluator.metrics)
        else:
            for m in metrics:
                assert m in self.evaluator.metrics
            self.set_metrics(metrics)
        task = specs_dict['task']
        if task == []:
            self.set_task([1 for i in range(len(self.metrics))])
        else:
            assert len(task) == len(self.metrics)
            self.set_task(task)


    def gencoordinates(self, m, n, j, k):
        '''Generate random coordinates in range x: (m,n) y:(j,k)

        instantiate generator and call next(g)

        based on:
        https://stackoverflow.com/questions/30890434/how-to-generate-random-pairs-of-
        numbers-in-python-including-pairs-with-one-entr
        '''
        seen = self.occupied
        x, y = randint(m, n), randint(j, k)
        while True:
            while (x, y) in seen:
                x, y = randint(m, n), randint(m, n)
            seen.add((x, y))
            yield (x, y)
        return

    def set_metrics(self, metrics):
        '''Define an array of metric names
        '''
        self.metrics = metrics

    def set_task(self, task):
        self.reward_weights = task

    def reset(self, initial=None, n_districts=8, max_blocks_per_district=5):
        '''Initialize the state randomly.
        '''
        if initial is not None:
            self.state = initial
            self.occupied = set(itertools.chain(*self.state.values()))
            return self.state

        else:
            self.occupied = set()
            self.state = {}
            # Place one block for each district, randomly
            for i in range(n_districts):
                self.state[i] = [next(self.coord_generator)]
            initial_blocks = [p[0] for p in self.state.values()]

            # add more blocks...
            for i in range(n_districts):
                # generate at most max_blocks_per_district new blocks per district
                # district_blocks = set(self.state[i])
                district_centroid = self.state[i][0]
                other_blocks = np.array(initial_blocks[:i] + [(float('inf'), float('inf'))] + initial_blocks[i + 1:])
                # distances = np.sqrt(np.sum(np.square(other_blocks - district_centroid), axis=1))
                distances = np.linalg.norm(other_blocks - district_centroid, axis=1)
                assert len(distances) == len(other_blocks)
                closest_pt_idx = np.argmin(distances)
                # closest_pt = other_blocks[closest_pt_idx]
                max_radius = distances[closest_pt_idx]/2
                for j in range(max(0, randint(0, max_blocks_per_district-1))):
                    dist = np.random.uniform(0, max_radius)
                    angle = np.random.uniform(0,2*np.pi)
                    new_block = district_centroid + np.array((dist*np.cos(angle),dist*np.sin(angle)))
                    new_block_coords = (new_block[0], new_block[1])
                    max_tries = 10
                    tries = 0
                    while new_block_coords in self.occupied and tries < max_tries:
                        tries += 1
                        dist = np.random.uniform(0, max_radius)
                        angle = np.random.uniform(0, 2 * np.pi)
                        new_block = district_centroid + (dist * np.cos(angle), dist * np.sin(angle))
                        new_block_coords = (int(new_block[0]), int(new_block[1]))
                    if tries < max_tries:
                        self.state[i].append(new_block_coords)
                        self.occupied.add(new_block_coords)

            return self.state

    def get_neighborhood(self, n_steps):
        '''Get all the configs that have one block n_steps away from the current
        '''
        neighborhood = []
        state = self.state
        for district_id, district in state.items():
            for block_id, block in enumerate(district):
                neighborhood += self.get_neighbors(district_id, block_id)
        return neighborhood

    def get_sampled_neighborhood(self, n_blocks, n_directions, resample=False):
        '''Sample n_blocks * n_direction neighbors.

        take n blocks, and move each one according to m direction/angle pairs
        ignore samples that are prima facie invalid (out of bounds or overlaps)
        if resample is true, then sample until we have n_blocks * n_directions
        otherwise, just try that many times.
        '''
        neighbors = []
        n_districts = len(self.state)
        for i in range(n_blocks):
            # sample from districts, then blocks
            # this biases blocks in districts with fewer blocks
            # i think this is similar to how humans work however
            district_id = np.random.randint(n_districts)
            district = self.state[district_id]
            block_id = np.random.randint(len(district))
            x,y = district[block_id]
            for j in range(n_directions):
                mx,my = self.get_random_move(x,y)
                valid_move = self.check_boundaries(mx,my) and (mx,my) not in self.occupied
                if valid_move:
                    neighbor = {k: list(val) for k, val in self.state.items()}
                    neighbor[district_id][block_id] = (mx, my)
                    neighbors.append(neighbor)
                elif resample == True:
                    # don't use this yet, need to add a max_tries?
                    while not valid_move:
                        mx,my = self.get_random_move(x,y)
                        valid_move = self.check_boundaries(mx,my)
        return neighbors

    def get_random_move(self, x, y):
        dist,angle = (np.random.randint(self.step_min, self.step_max),
                        np.random.uniform(2*np.pi))
        return (int(x + np.cos(angle) * dist), int(y + np.sin(angle) * dist))

    def check_boundaries(self, x, y):
        '''Return true if inside screen boundaries
        '''
        if x < self.x_min or x > self.x_max:
            return False
        if y < self.y_min or y > self.y_max:
            return False
        return True

    def get_neighbors(self, district, block):
        '''Get all the designs that move "block" by one step.


        ignores moves to coords that are occupied or out of bounds
        '''
        neighbors = []

        moves = [np.array((self.step, 0)), np.array((-self.step, 0)),
                 np.array((0, self.step)), np.array((0, -self.step))]

        constraints = [lambda x, y: x < self.x_max,
                        lambda x, y: x > self.x_min,
                        lambda x, y: y < self.y_max,
                        lambda x, y: y > self.y_min]

        x, y = self.state[district][block]

        for i, move in enumerate(moves):
            mx, my = (x, y) + move
            if constraints[i](mx, my) and (mx, my) not in self.occupied:
                new_neighbor = deepcopy(self.state)
                new_neighbor[district][block] = (mx, my)
                neighbors.append(new_neighbor)

        return neighbors

    def check_legal_districts(self, districts):
        if len(districts) == 0:
            return False
        # TODO: consider checking for len == 8 here as well
        for d in districts:
            if len(d.precincts) == 0:
                return False
        return True

    def get_metrics(self, design, exc_logger=None):
        '''Get the vector of metrics associated with a design

        returns m-length np array
        '''
        try:
            districts = self.evaluator.get_voronoi_districts(design)
            state_metrics, districts = self.evaluator.compute_voronoi_metrics(districts)
        except ColliderException:
            if exc_logger is not None:
                exc_logger.write(str(design) + '\n')
            else:
                print("Collider Exception!")
            return None

        if not self.check_legal_districts(districts):
            return None

        metric_dict = {}
        for state_metric in state_metrics:
            metric_name = state_metric.name
            if metric_name in self.metrics:
                metric_dict[metric_name] = self.metric_extractors[metric_name](state_metric, districts)

        metrics = np.array([metric_dict[metric] for metric in self.metrics])
        #metrics = np.array([self.metric_extractors[metric](state_metrics, districts) for metric in self.metrics])
        return metrics

    def get_reward(self, metrics):
        '''Get the scalar reward associated with metrics
        '''
        if metrics is None:
            return float("-inf")
        else:
            return np.dot(self.reward_weights, self.standardize_metrics(metrics))

    def standardize_metrics(self, metrics):
        '''Standardizes the metrics if standardization stats have been provided.
        '''
        if self.pop_mean is None or self.pop_std is None:
            return metrics
        else:
            return (metrics - self.pop_mean)/self.pop_std

    def run(self, environment, n_steps, logger=None, exc_logger=None, status=None, initial=None, eps=0.8, eps_decay=0.9,
            eps_min=0.1, n_tries_per_step = 10):
        '''runs for n_steps and returns traces of designs and metrics
        '''
        self.reset(initial)
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
        while i < n_steps:
            i += 1
            if i % 50 == 0:
                last_reward = float("-inf")
                self.reset(initial)
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
                neighborhood = self.get_sampled_neighborhood(4,2)
                if len(neighborhood) < 1:
                    continue
                else:
                    metrics = [self.get_metrics(n, exc_logger) for n in neighborhood]
                    count += len(metrics)
                    rewards = [self.get_reward(m) for m in metrics]
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
                    self.reset(initial)
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
            self.state = neighborhood[best_idx]
            self.occupied = set(itertools.chain(*self.state.values()))
            if status is not None:
                status.put('next')
            if logger is not None:
                logger.write(str([time.perf_counter(), count, list(metrics[best_idx]), self.state]) + '\n')
            else:
                mappend(metrics[best_idx])
                dappend(self.state)
        if logger is not None:
            return "n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights
        else:
            print("n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms), self.reward_weights)
            return design_log,metric_log


# if __name__ == '__main__':
# #    ga = GreedyAgent(metrics=['population','pvi','compactness','projected_votes','race','income','area'])
#     ga = GreedyAgent(metrics=['population','pvi','compactness','projected_votes','race'])
# #    ga.set_task([0,0,0,1,0,0,0])
#     ga.set_task([0,0,0,1,0])
#     print(ga.reset())
#     designs, metrics = ga.run(2)
# class GreedyAgent(Agent):
#     def __init__(self):
#         pass
#     def run(self,environment,n_steps):
#         print("Running the Greedy Agent in {} for {} steps.".format(environment,n_steps))
