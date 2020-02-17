from random import randint
import itertools
import numpy as np
from copy import deepcopy
import pickle
from environments import Environment
from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException


class DistopiaDQNEnvironment(Environment):
    # want to define the voronoi agent for districts and metrics calculations
    # scalar_value is mean over districts
    # scalar_std is standard deviation between districts
    # scalar_maximum is max over districts
    # s is state metric object (for this metric)
    # d is list of district objects (for all metrics)
    metric_extractors = {

        # overall normalization plan: run one-hots in either direction to get rough bounds
        # then z-normalize and trim on edges
        # 'population' : lambda s,d : s.scalar_std,
        # standard deviation of each district's total populations (-1)
        # normalization: [0, single_district std]
        'population': lambda s, d: np.std([dm.metrics['population'].scalar_value for dm in d]),
        # mean of district margin of victories (-1)
        # normalization: [0,1]
        'pvi': lambda s, d: s.scalar_maximum,
        # minimum compactness among districts (maximize the minimum compactness, penalize non-compactness) (+1)
        # normalization: [0,1]
        'compactness': lambda s, d: np.mean([dm.metrics['compactness'].scalar_value for dm in d]),
        # mean ratio of democrats over all voters in each district (could go either way)
        # normalization: [0,1]
        'projected_votes': lambda s, d: np.mean(
            [dm.metrics['projected_votes'].scalar_value / dm.metrics['projected_votes'].scalar_maximum for dm in d]),
        # std of ratio of nonminority to minority over districts
        # normalization: [0, ]
        'race': lambda s, d: np.std([dm.metrics['race'].scalar_value / dm.metrics['race'].scalar_maximum for dm in d]),
        # scalar value is std of counties within each district. we take a max (-1) to minimize variance within district (communities of interest)
        'income': lambda s, d: np.max([dm.metrics['income'].scalar_value for dm in d]),
        # 'education' : lambda s,d : s.scalar_std,

        # maximum sized district (-1) to minimize difficulty of access
        # normalization [0,size of wisconsin]
        'area': lambda s, d: s.scalar_maximum
    }
    json_metric_extractors = {

        # overall normalization plan: run one-hots in either direction to get rough bounds
        # then z-normalize and trim on edges
        # 'population' : lambda s,d : s.scalar_std,
        # standard deviation of each district's total populations (-1)
        # normalization: [0, single_district std]
        #'population': lambda s, d: np.std([dm['metrics']['population']['scalar_value'] for dm in d]),
        'population': lambda s, d: np.std([[m['scalar_value'] for m in dm['metrics'] if m['name'] == 'population'][0] for dm in d]),
        # mean of district margin of victories (-1)
        # normalization: [0,1]
        'pvi': lambda s, d: s['scalar_maximum'],
        # minimum compactness among districts (maximize the minimum compactness, penalize non-compactness) (+1)
        # normalization: [0,1]
        #'compactness': lambda s, d: np.min([dm['metrics']['compactness']['scalar_value'] for dm in d]),
        'compactness': lambda s, d: np.mean([[m['scalar_value'] for m in dm['metrics'] if m['name'] == 'compactness'][0] for dm in d]),
        # TODO: change compactness from min to avg
        # mean ratio of democrats over all voters in each district (could go either way)
        # normalization: [0,1]
        'projected_votes': lambda s, d: np.mean([[m['scalar_value']/m['scalar_maximum'] for m in dm['metrics'] if m['name'] == 'projected_votes'][0] for dm in d]),
        #[dm['metrics']['projected_votes']['scalar_value'] / dm['metrics']['projected_votes']['scalar_maximum'] for dm in d]),
        # std of ratio of nonminority to minority over districts
        # normalization: [0, ]
        'race': lambda s, d: np.std([[m['scalar_value'] for m in dm['metrics'] if m['name'] == 'race'][0] for dm in d]),
        #lambda s, d: np.std([dm['metrics']['race']['scalar_value'] / dm['metrics']['race']['scalar_maximum'] for dm in d]),
        # scalar value is std of counties within each district. we take a max (-1) to minimize variance within district (communities of interest)
        'income': lambda s, d: np.max([dm['metrics']['income']['scalar_value'] for dm in d]),
        # 'education' : lambda s,d : s.scalar_std,

        # maximum sized district (-1) to minimize difficulty of access
        # normalization [0,size of wisconsin]
        'area': lambda s, d: s['scalar_maximum']
    }
    def __init__(self, x_lim=(100, 900), y_lim=(100, 900),
                  step_min=10, step_max=100,
                 pop_mean=None, pop_std=None):
        print('initializing DistopiaEnvironment')
        self.x_min, self.x_max = x_lim
        self.y_min, self.y_max = y_lim
        self.step = 1
        self.step_min = step_min
        self.step_max = step_max
        self.pop_mean = pop_mean
        self.pop_std = pop_std
        self.occupied = set()
        self.coord_generator = self.gencoordinates(self.x_min, self.x_max, self.y_min, self.y_max)
        self.evaluator = VoronoiAgent()
        self.evaluator.load_data()
        self.state = {}
        self.mean_array = self.std_array = None
        self.subsample_scale = 1  # defaults to 1->no subsampling.
        # Usage of subsampling: if subsample scale of n, then starting from xmin and ymin,
        # block positions can only be at coordinates at n pixels intervals
        assert self.step%self.subsample_scale == 0
        assert self.step>=self.subsample_scale

    def set_normalization(self, standard_file, st_metrics):
        #Only reason this is a seperate method is so that I can instantiate distopia_environment
        #and set up normalization without doing anything else in distopia_human_logs_processor
        with open(standard_file, 'rb') as f:
            self.mean_array, self.std_array = pickle.load(f)
            if type(self.mean_array) is not np.ndarray:
                self.mean_array = np.array(self.mean_array)
            if type(self.std_array) is not np.ndarray:
                self.std_array = np.array(self.std_array)
            # cut down the metrics to the ones that we are using.
            # this is not going to work if they are out of order
            # TODO: generalize this, maybe by adding metadata to the data file
            self.mean_array = self.mean_array[:len(st_metrics)]
            self.std_array = self.std_array[:len(st_metrics)]

    def set_params(self, specs_dict):
        metrics = specs_dict['metrics']
        if metrics == []:
            self.set_metrics(self.evaluator.metrics)
        else:
            for m in metrics:
                assert m in self.evaluator.metrics
            self.set_metrics(metrics)
        if 'standardization_file' in specs_dict and specs_dict['standardization_file'] is not None:
            # hopefully the above condition short-circuits
            self.set_normalization(specs_dict['standardization_file'], self.metrics)
        if 'subsample_scale' in specs_dict:
            self.subsample_scale = specs_dict['subsample_scale']
            self.step = self.subsample_scale
            assert self.step%self.subsample_scale == 0
            assert self.step>=self.subsample_scale
            print("subsample scale environment: " + str(self.subsample_scale))


    def gencoordinates(self, m, n, j, k):
        '''Generate random coordinates in range x: (m,n) y:(j,k)
        instantiate generator and call next(g)

        based on:
        https://stackoverflow.com/questions/30890434/how-to-generate-random-pairs-of-
        numbers-in-python-including-pairs-with-one-entr

        MODIFIED: To handle the case where the map is subsampled to reduce state space
        for the SARSA agent
        '''
        seen = self.occupied
        x_range = np.arange(m, n+1, self.subsample_scale)
        y_range = np.arange(j, k+1, self.subsample_scale)
        # print("IN GENCOORDINATES, THE XRANGE IS ", x_range)
        x, y = np.random.choice(x_range), np.random.choice(y_range)
        while True:
            while (x, y) in seen:
                x, y = np.random.choice(x_range), np.random.choice(y_range)
            seen.add((x, y))
            yield (x, y)
        return

    def set_metrics(self, metrics):
        '''Define an array of metric names
        '''
        self.metrics = metrics

    def seed(self,seed):
        np.random.seed(seed)

    def take_step(self, new_state):
        self.state = new_state
        self.occupied = set(itertools.chain(*self.state.values()))


    def reset(self, initial=None, n_districts=8, max_blocks_per_district=5):
        '''Initialize the state randomly.
        '''
        #Generator must be reset bc environment is reset at every episode
        self.coord_generator = self.gencoordinates(self.x_min, self.x_max, self.y_min, self.y_max)
        if initial is not None:
            self.state = initial
            self.occupied = set(itertools.chain(*self.state.values()))
            return self.state

        else:
            # print("resetting to initial state....")
            while True:
                self.occupied = set()
                self.state = {}
                # Place one block for each district, randomly
                for i in range(n_districts):
                    self.state[i] = [next(self.coord_generator)]
                initial_blocks = [p[0] for p in self.state.values()]

                # add more blocks...
                # for i in range(n_districts):
                #     # generate at most max_blocks_per_district new blocks per district
                #     # district_blocks = set(self.state[i])
                #     district_centroid = self.state[i][0]
                #     other_blocks = np.array(initial_blocks[:i] + [(float('inf'), float('inf'))] + initial_blocks[i + 1:])
                #     # distances = np.sqrt(np.sum(np.square(other_blocks - district_centroid), axis=1))
                #     distances = np.linalg.norm(other_blocks - district_centroid, axis=1)
                #     assert len(distances) == len(other_blocks)
                #     closest_pt_idx = np.argmin(distances)
                #     # closest_pt = other_blocks[closest_pt_idx]
                #     max_radius = distances[closest_pt_idx]/2
                #     for j in range(max(0, randint(0, max_blocks_per_district-1))):
                #         dist = np.random.uniform(0, max_radius)
                #         angle = np.random.uniform(0,2*np.pi)
                #         new_block = district_centroid + np.array((dist*np.cos(angle),dist*np.sin(angle)))
                #         new_block_coords = (new_block[0], new_block[1])
                #         max_tries = 10
                #         tries = 0
                #         while new_block_coords in self.occupied and tries < max_tries:
                #             tries += 1
                #             dist = np.random.uniform(0, max_radius)
                #             angle = np.random.uniform(0, 2 * np.pi)
                #             new_block = district_centroid + (dist * np.cos(angle), dist * np.sin(angle))
                #             new_block_coords = (int(new_block[0]), int(new_block[1]))
                #         if tries < max_tries:
                #             self.state[i].append(new_block_coords)
                #             self.occupied.add(new_block_coords)
                if self.get_metrics(self.state) is not None:
                    return self.state
            # print("initial state resetted, printing the current positions")
            # print(self.state)


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

    def make_move(self, block_to_move, direction):
        """Moves the specified block in the specified direction, return the new design"""

        moves = [np.array((self.step, 0)), np.array((-self.step, 0)),
                 np.array((0, self.step)), np.array((0, -self.step))]
        constraints = [lambda x, y: x < self.x_max,
                        lambda x, y: x > self.x_min,
                        lambda x, y: y < self.y_max,
                        lambda x, y: y > self.y_min]
        move = moves[direction]
        x, y = self.state[block_to_move][0] # here assuming each district only holds one block
        mx, my = (x, y) + move
        if constraints[direction](mx, my) and (mx, my) not in self.occupied:
            new_state = deepcopy(self.state)
            new_state[block_to_move][0] = (mx, my)
            return new_state
        else:
            return -1

    def get_boundaries(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max]

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
        except AttributeError:
            print("FAILED DESIGN IS: ", design)
        except AssertionError as e:
            if exc_logger is not None:
                exc_logger.write(str(design) + '\n')
            else:
                print("Assertion failed: {}".format(e.args))
            return None

        if not self.check_legal_districts(districts):
            return None
        return self.extract_metrics(self.metrics,state_metrics,districts)

    def get_district_metrics(self, design, exc_logger=None):
        """returns the metrics associated with each district"""
        try:
            districts = self.evaluator.get_voronoi_districts(design)
            state_metrics, _ = self.evaluator.compute_voronoi_metrics(districts)
            unpacked_metrics = {}
            for i, d in enumerate(districts):
                unpacked_metrics[i] = {}
                unpacked_metrics[i][0] = d.metrics['population'].scalar_value
                unpacked_metrics[i][1] = d.metrics['pvi'].scalar_value
                unpacked_metrics[i][2] = d.metrics['compactness'].scalar_value
            return unpacked_metrics
        except ColliderException:
            if exc_logger is not None:
                exc_logger.write(str(design) + '\n')
            else:
                print("Collider Exception!")
            return None

    @staticmethod
    def extract_metrics(metric_names,state_metrics,districts,from_json=False):
        metric_dict = dict()
        for state_metric in state_metrics:
            if from_json:
                metric_name = state_metric["name"]
                if metric_name in metric_names:
                    metric_dict[metric_name] = DistopiaDQNEnvironment.json_metric_extractors[metric_name](state_metric, districts)
            else:
                metric_name = state_metric.name
                if metric_name in metric_names:
                    metric_dict[metric_name] = DistopiaDQNEnvironment.metric_extractors[metric_name](state_metric, districts)

        metrics = np.array([metric_dict[metric] for metric in metric_names])
        return metrics
    def get_reward(self, metrics, reward_weights):
        '''Get the scalar reward associated with metrics
        '''
        if metrics is None:
            return float("-inf")
        else:
            return np.dot(reward_weights, self.standardize_metrics(metrics))

    def standardize_metrics(self, metrics):
        '''Standardizes the metrics if standardization stats have been provided.
        '''
        if self.mean_array is None or self.std_array is None:
            return metrics
        else:
            if type(metrics) is not np.ndarray:
                metrics = np.array(metrics)
            return (metrics - self.mean_array)/self.std_array

    def destandardize_metrics(self, metrics):
        '''Undo's standardization
        '''
        if self.mean_array is None or self.std_array is None:
            return metrics
        else:
            if type(metrics) is not np.ndarray:
                metrics = np.array(metrics)
            return metrics*self.std_array + self.mean_array

    def fixed2dict(self, fixed_arr):
        '''Convert a fixed array of nx8 to an 8 district dict
            The fixed_arr should be in form [x0,y0,x1,y1...]
            Strips out zeros
        '''
        dist_dict = dict()
        assert len(fixed_arr) % 8 == 0
        blocks_per_dist = len(fixed_arr)//8

        for i in range(len(fixed_arr),2):
            x = fixed_arr[i]
            y = fixed_arr[i+1]
            district = i//blocks_per_dist
            if district in dist_dict:
                dist_dict[district].append((x,y))
            else:
                dist_dict[district] = [(x,y)]
        return dist_dict

    def dict2fixed(self, dist_dict, blocks_per_dist):
        '''Convert a district dict into a fixed-width array (zero-padded)
        '''
        fixed = []
        for district, blocks in dist_dict.items():
            n_to_pad = blocks_per_dist - len(blocks)
            assert n_to_pad >= 0
            for block in blocks:
                fixed.append(block[0])
                fixed.append(block[1])
            for i in range(n_to_pad):
                fixed.append(0.0) # x
                fixed.append(0.0) # y
        return fixed
