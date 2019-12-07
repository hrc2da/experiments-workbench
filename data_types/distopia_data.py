from data_types import Data
from distopia.app.agent import VoronoiAgent
from utils import hierarchical_sort
import numpy as np
import pickle as pkl
import csv
from tqdm import tqdm
from multiprocessing import Pool, Manager
from threading import Thread
from environments.distopia_environment import DistopiaEnvironment
import itertools
import ast


class DistopiaData(Data):
    master_metric_list = ['population', 'pvi', 'compactness','projected_votes','race']
    def __init__(self):
        self.voronoi = VoronoiAgent()
        self.voronoi.load_data()
    def set_params(self, specs):
        for param, param_val in specs.items():
            setattr(self, param, param_val)
        self.preprocessors = specs["preprocessors"]
        self.generate_task_dicts(len(self.metric_names))
        # if "n_workers" in specs:
        #     self.n_workers = specs["n_workers"]
        # else:
        #     self.n_workers = 1

    def load_agent_data(self, fname,fmt=None, labels_path=None, append=False, load_designs=False, load_metrics=False,
        load_rewards=False, norm_file=None, data_dir=None):
        """Loads the log file from running agent
            Assumes that the log file contains data from multiple tasks"""
        env = DistopiaEnvironment()  # TODO: This is a temp fix to standardize human metrics
        env.set_normalization(norm_file, self.metric_names)
        logs = self.load_json(fname)
        cur_task = None
        task_counter = 0
        cur_trajectory = []
        trajectories = []

#        for log in logs:
        rewards = []
        cur_task = logs[0]["task"]
        print(cur_task)
        cur_trajectory = []
        task_counter += 1
        for episode in logs[0]['episodes']:
            for step in episode['run_log']:
                step_tuple = []
                if load_designs:
                    step_districts = self.jsondistricts2mat(step['design'])
                    step_tuple.append(step_districts)
                if load_metrics:
                    assert hasattr(self, "metric_names")
                    step_metrics = env.standardize_metrics(self.task_str2arr(step['metrics']))
                    step_tuple.append(step_metrics)
                if load_rewards:
                    rewards.append(step['reward'])
                cur_trajectory.append(step_tuple)
        if load_rewards:
            self.rewards = rewards
            if data_dir:
                self.save_rewards(data_dir, fname)
            else:
                self.save_rewards('', fname)
            trajectories.append((cur_trajectory[:], cur_task))
        if append == False or not hasattr(self, 'x') or not hasattr(self, 'y'):
            self.y = []
            self.x = []
        else:
            self.y = list(self.y)
            self.x = list(self.x)
        x = []
        y = []
        for trajectory in trajectories:
            samples, task = trajectory
            for sstep in samples:
                x.append(*sstep)  # any sample data
                y.append(task)  # task

        for preprocessor in self.preprocessors:
                x,y = getattr(self,preprocessor)((x,y))
        for i in x:
            self.x.append(i)
        for j in y:
            self.y.append(j)

        self.x = np.array(self.x)
        self.y = np.array(self.y)


    def load_data(self, fname, fmt=None, labels_path=None, append=False, load_designs=False, load_metrics=False, norm_file = None, load_fiducials=False):
        if fmt is None:
            fmt = self.infer_fmt(fname)
        print(fmt)
        if fmt == "pkl" or fmt== "pk":
            metrics,designs = self.load_pickle(fname)
            if not hasattr(self,'feature_type'):
                raw_data = designs
            elif self.feature_type == "metrics":
                raw_data = self.taskdict2vect(metrics)
            elif self.feature_type == "designs":
                raw_data = designs
            self.x,self.y = raw_data
            for preprocessor in self.preprocessors:
                self.x,self.y = getattr(self,preprocessor)((self.x,self.y)) #design_dict2mat_labelled(raw_data)

            # try to force de-allocation
            raw_data = None
        elif fmt == "npy":
            assert labels_path is not None
            self.x = np.load(fname)
            self.y = np.load(labels_path)
            for preprocessor in self.preprocessors:
                self.x,self.y = getattr(self,preprocessor)((self.x,self.y))
        elif fmt == "json": #it's a log file (for now)
            # env = DistopiaEnvironment()  # TODO: This is a temp fix to standardize human metrics
            # env.set_normalization(norm_file, self.metric_names)
            logs = self.load_json(fname)
            trajectories = [] # tuple of trajectory, focus_trajectory, and label
            cur_task = None
            cur_focus = "None"
            cur_trajectory_districts = []
            cur_trajectory_metrics = []
            cur_trajectory = []
            task_counter = 0
            for log in logs:
                keys = log.keys()
                step_tuple = []
                if "task" in keys:
                        trajectories.append((cur_trajectory[:],cur_task))
                        cur_task = log["task"]
                        print(cur_task)
                        cur_trajectory = []
                        task_counter += 1
                elif "focus" in keys and log['focus']['cmd'] == 'focus_state':
                        cur_focus = log['focus']['param']
                elif cur_task is None:
                    continue
                elif "districts" in keys:
                    if len(log['districts']['districts']) < 8:
                        continue
                    district_sizes = [len(d['precincts']) for d in log['districts']['districts']]
                    if min(district_sizes) < 1:
                        continue
                    if load_designs == True:
                        step_districts = self.jsondistricts2mat(log['districts']['districts'])
                        step_tuple.append(step_districts)
                    if load_metrics == True:
                        assert hasattr(self,"metric_names")
                        #perform normalization on human data
                        # step_metrics = env.standardize_metrics(DistopiaEnvironment.extract_metrics(self.metric_names,log['districts']['metrics'],
                        #                                         log['districts']['districts'],from_json=True))
                        step_metrics = DistopiaEnvironment.extract_metrics(self.metric_names,log['districts']['metrics'],
                                                                log['districts']['districts'],from_json=True)
                        step_tuple.append(step_metrics)
                    cur_trajectory.append(step_tuple)
            trajectories.append((cur_trajectory[:],cur_task))
            if append == False or not hasattr(self,'x') or not hasattr(self,'y'):
                self.y = []
                self.x = []
            else:
                self.y = list(self.y)
                self.x = list(self.x)
            x = []
            y = []
            for trajectory in trajectories:
                samples,task = trajectory
                for sstep in samples:
                    x.append(*sstep) #any sample data
                    y.append(task) #task
            for preprocessor in self.preprocessors:
                x,y = getattr(self,preprocessor)((x,y))
            #TODO: make this more efficient. probably use np concat or something
            for i in x:
                self.x.append(i)
            for j in y:
                self.y.append(j)
            self.x = np.array(self.x)
            self.y = np.array(self.y)



        elif fmt == "csv":
            # TODO: no pre-processing for now, let's fix this later.
            assert labels_path is not None
            raw_x = self.load_csv(fname)
            raw_y = self.load_csv(labels_path)
            for preprocessor in self.preprocessors:
                raw_x,raw_y = getattr(self,preprocessor)((raw_x,raw_y))

            if append == False or not hasattr(self,'x') or not hasattr(self,'y'):
                self.x = raw_x
                self.y = raw_y
            else:
                self.x = np.concatenate((self.x, raw_x))
                self.y = np.concatenate((self.y, raw_y))
    def generate_task_dicts(self,dim):
        if not hasattr(self, "task_labels"):
            self.task_labels = hierarchical_sort(list(map(np.array,itertools.product(*[[-1., 0., 1.]]*dim))))
        else:
            self.task_labels = hierarchical_sort(list(map(np.array,self.task_labels)))
        self.task_ids = {self.task_arr2str(task) : i for i,task in enumerate(self.task_labels)}
        self.task_dict = {self.task_arr2str(task) : task for task in self.task_labels}
    # pre-processing functions
    def save_csv(self,xfname,yfname):
        print(xfname)
        with open(xfname+".csv", 'w+', newline='') as samplefile:
            with open(yfname+"_labels.csv", 'w+', newline='') as labelfile:
                samplewriter = csv.writer(samplefile)
                labelwriter = csv.writer(labelfile)
                for i,sample in enumerate(self.x):
                    samplewriter.writerow(sample.flatten())
                    labelwriter.writerow(self.y[i])

    def save_rewards(self, data_dir, fname):
        with open(str(fname)+'_rewards_pickle.txt', 'wb') as rewardsfile:
            print(rewardsfile)
            pkl.dump(self.rewards, rewardsfile)

    def save_npy(self,xfname,yfname):
        np.save(xfname, self.x)
        np.save(yfname, self.y)

    def standardize(self,data,standardization_file=None):
        env = DistopiaEnvironment()
        if standardization_file is None:
            if self.standardization_file is None:
                raise ValueError("No standardization params are set!")
            else:
                standardization_file = self.standardization_file

        env.set_normalization(standardization_file, self.metric_names)

        x,y = data
        return env.standardize_metrics(x),y

    def destandardize(self,data,standardization_file=None):
        env = DistopiaEnvironment()
        if standardization_file is None:
            if self.standardization_file is None:
                raise ValueError("No standardization params are set!")
            else:
                standardization_file = self.standardization_file

        env.set_normalization(standardization_file, self.metric_names)

        x,y = data
        return env.destandardize_metrics(x),y

    # pre-process labels
    def filter_by_task(self,data):
        x,y = data
        labels = list(self.task_dict.keys())
        mask = [self.task_arr2str(label) in labels for label in y]

        return x[mask],y[mask]
    def slice_metrics_to_3(self,data):
        x,y = data
        return x[:,:3],y
    def slice_metrics_to_4(self,data):
        x,y = data
        return x[:,:4],y
    def onehot2class(self,data):
        x,y = data
        y_out = []
        for label in y:
            y_out.append(self.task_ids[self.task_arr2str(label)])
        print(y_out[:10])
        return x,y_out

    def class2onehot(self,data):
        x,y = data
        y_out = []
        for label in y:
            y_out.append(self.task_labels[label])
        return x,np.array(y_out)

    @staticmethod
    def unflatten_districts(data):
        flattened_arr_list,labels = data
        n,flat_dim = flattened_arr_list.shape
        assert flat_dim == 72*8
        return flattened_arr_list.reshape(n,72,8),labels

    @staticmethod
    def task_str2arr(task_str):
        ''' convert a stringified np array back to the array
        :param string: the stringified np array
        :return: a np array
        '''
        assert type(task_str) == str
        assert task_str[0] == '['
        assert task_str[-1] == ']'
        if ',' in task_str:
            return np.array(eval(task_str))
        else:
            return np.array(task_str[1:-1].split(), dtype=float)

    @staticmethod
    def task_arr2str(arr):
        return str(np.array(arr, dtype=float))
    # pre-process designs
    def truncate_design_dict(self,design_dict):
        assert(hasattr(self,"slice_lims"))
        start,limit = self.slice_lims
        for key,samples in design_dict.items():
            design_dict[key] = samples[start:limit]
        return design_dict


    def filter_by_metrics(self,data):
        '''remove all the data with labels that have nonzero weight on metrics that are not on our list
        '''
        x,y = data
        x_out = []
        y_out = []
        metric_indices = [self.master_metric_list.index(metric) for metric in self.metric_names]
        for i, label in enumerate(y):
            in_scope = True
            for j, weight in enumerate(label):
                if np.abs(weight) == 1 and self.master_metric_list[j] not in self.metric_names:
                    in_scope = False
                    break
            if in_scope == True:
                x_out.append(x[i,:])
                y_out.append(label[metric_indices])
        return np.array(x_out),np.array(y_out)


    def sliding_window(self,data,window_size=40):
        if hasattr(self,"window_size"):
            window_size = self.window_size
        # should probably also check if metrics or designs here
        if type(data) == dict:
            # chunk into 50 and slide the window
            task_dict = dict()
            for key,val in data.items():
                upper_bound = val.shape[0] - val.shape[0]%50
                truncated = val[:upper_bound]
                n_steps, metric_dim = truncated.shape
                n_chunks = n_steps // 50
                task_dict[key] = val.reshape(n_chunks,50,metric_dim)
        else:
            x,y = data
            x_out = []
            y_out = []
            task_dict = self.get_task_dict(x,y,merge=False)
        for key,val in task_dict.items():
            for instance in val:
                #slide across as much as possible
                i = 0
                while i + window_size < len(instance):
                    x_win = (instance[i:i+window_size])
                    x_out.append(instance[i:i+window_size])
                    y_out.append(self.task_str2arr(key))
                    i += 1
        return np.array(x_out),np.array(y_out)
    def balance_samples(self,data):
        '''cuts samples to sample size for each class.
        Assumes that the order of the data doesn't matter.
        '''
        x,y = data
        labels = set(y)
        x_ = None
        y_ = None
        y = np.array(y)
        for label in labels:
            if x_ is None:
                x_ = x[np.random.choice(np.where(y == label)[0],self.balanced_sample_size)]
                assert y_ is None
                y_ = [label for i in range(self.balanced_sample_size)]
            else:
                assert len(x_) == len(y_)
                x_ = np.concatenate([x_,x[np.random.choice(np.where(y == label)[0],self.balanced_sample_size)]])
                y_ = np.concatenate([y_,[label for i in range(self.balanced_sample_size)]])
        return x_,y_





    def conv3dreshape(self,data):
        x,y = data
        n,w,h,d = x.shape
        return x.reshape(n,h,d,w,1),y

    def strip_repeats(self,data):
        x,y = data
        x_out = [x[0]]
        y_out = [y[0]]
        last_sample = x[0]
        for i,sample in enumerate(x[1:],1):
            if not np.array_equal(last_sample,sample):
                x_out.append(sample)
                y_out.append(y[i])
                last_sample = sample
        return np.array(x_out),np.array(y_out)
    @staticmethod
    def window_stack(a,stepsize=1,width=3):
        return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )
    def sliding_window_arr(self,data):
        print("sliding window")
        assert hasattr(self,"window_step")
        assert hasattr(self,"window_size")
        # this is expensive, but the easiest way I think of to do this is to make a dict on label, convert to windows, and recreate the array
        data_dict = dict()
        x_arr,y_arr = data
        for i,x in enumerate(x_arr):
            y = y_arr[i]
            y = self.task_arr2str(y)
            if y in data_dict:
                data_dict[y].append(x)
            else:
                data_dict[y] = [x]
        windowed_x = None
        windowed_y = None
        for y,xs in data_dict.items():
            wxs = self.window_stack(np.array(xs),self.window_step,self.window_size)
            if windowed_x is None:
                windowed_x = [wxs]
                windowed_y = [self.task_str2arr(y)]
            else:
                import pdb; pdb.set_trace()
                windowed_x = np.concatenate([windowed_x,wxs])
                windowed_y = np.concatenate([windowed_y, self.task_str2arr(y)])

        return windowed_x,windowed_y


        for key,samples in design_dict.items():
            design_dict[key] = self.window_stack(samples,self.window_step,self.window_size)
        return design_dict

    def sliding_window_dict(self,design_dict):
        print("sliding window")
        assert hasattr(self,window_step)
        assert hasattr(self,window_size)
        import pdb; pdb.set_trace()
        for key,samples in design_dict.items():
            design_dict[key] = self.window_stack(samples,self.window_step,self.window_size)
        return design_dict

    def design_dict2mat_labelled(self,design_dict):
        print("Converting designs to matrices.")
        lengths = [len(samples) for samples in design_dict.values()]
        n_samples = np.sum(lengths)
        print("allocating space.")

        print("converting {} designs.".format(n_samples))
        if self.n_workers < 2:
            x = np.zeros((n_samples,72,8),dtype=np.uint8)
            y = np.zeros((n_samples,5),dtype=np.uint8)
            sample_counter = 0
            with tqdm(total=n_samples) as bar:
                for key in design_dict.keys():
                    for sample in design_dict[key]:
                        x[sample_counter,:,:] = self.fiducials2district_mat(sample)
                        y[sample_counter,:] = self.task_str2arr(key)
                        sample_counter += 1
                        bar.update(1)
        else:
            print("Multi-Processing the preprocessor")
            # get the start index for dict entry that we are running through here
            start_indices = [0]
            for length in lengths:
                start_indices.append(start_indices[-1] + length)
            progress_queue = Manager().Queue()
            progress_thread = Thread(target=self.progress_monitor,args=(n_samples,progress_queue))
            progress_thread.start()
            queued_tasks = [(design_dict[key],key, progress_queue) for i,key in enumerate(design_dict.keys())]

            with Pool(self.n_workers) as pool:
                results = pool.starmap(self.designs2mat_labelled_helper, queued_tasks)
            # block here until finished
            x,y = map(np.concatenate, zip(*results))
        assert(np.sum(x) == n_samples*72)
        return x,y

    @staticmethod
    def designs2mat_labelled_helper(designs,task,progress_queue):
        temp_voronoi = VoronoiAgent()
        temp_voronoi.load_data()
        x = np.zeros((len(designs),72,8))
        y = np.zeros((len(designs),5))
        for i,design in enumerate(designs):
            # should work, since it is pass-by-object-reference
            x[i,:,:] = DistopiaData.static_fiducials2district_mat(design, voronoi=temp_voronoi)
            y[i,:] = DistopiaData.task_str2arr(task)
            progress_queue.put(1)
        return x,y


    def progress_monitor(self,n_samples,progress_queue):
        for i in tqdm(range(n_samples)):
            progress_queue.get()

    @staticmethod
    def static_fiducials2district_mat(fiducials,voronoi):
        districts = voronoi.get_voronoi_districts(fiducials)
        district_mat = DistopiaData.districts2mat(districts)
        return district_mat

    def fiducials2district_mat(self,fiducials, voronoi=None):
        '''converts a dict of fiducials into a matrix representation of district assignments
           the matrix representation is 72x8 one-hot
        '''
        if voronoi is None:
            voronoi = self.voronoi
        districts = voronoi.get_voronoi_districts(fiducials)
        district_mat = self.districts2mat(districts)
        return district_mat

    @staticmethod
    def districts2mat(district_list):
        '''takes a list of district objects and returns an occupancy matrix
            of precincts by district (72x8 one-hot matrix where mat[a,b] indicates whether
            precinct a is in district b)
        '''
        mat = np.zeros((72,8), dtype=int)
        for i,district in enumerate(district_list):
            precincts = district.precincts
            for precinct in precincts:
                mat[int(precinct.identity),i] = 1
        return mat

    @staticmethod
    def jsondistricts2mat(district_list):
        '''takes a list of district objects and returns an occupancy matrix
            of precincts by district (72x8 one-hot matrix where mat[a,b] indicates whether
            precinct a is in district b)
        '''
        mat = np.zeros((72,8), dtype=int)
        for i,district in enumerate(district_list):
            precincts = district['precincts']
            for precinct in precincts:
                mat[int(precinct),i] = 1
        return mat

    def get_task_dict(self, x=None, y=None, merge=False):
        '''Get a dictionary of trajectories keyed on task
            if merge is True, then concat the trajectories
        '''
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        task_dict = dict()
        task_keys = set()
        # start by getting the list of tasks in the data
        for task in y:
            task_keys.add(self.task_arr2str(task))

        for key in task_keys:
            task_arr = self.task_str2arr(key)
            indices = np.where((y == task_arr).all(axis=1))[0]
            if merge:
                task_dict[key] = x[indices]
            # otherwise, we have to split the indices
            else:
                task_dict[key] = []
                last_idx = indices[0]
                start_idx = indices[0]
                for idx in indices[1:]:
                    if idx - last_idx > 1:
                        end_idx = last_idx
                        if end_idx-start_idx > 1:
                            task_dict[key].append(x[start_idx:end_idx])
                        start_idx = idx
                    last_idx = idx
                if start_idx < last_idx:
                    task_dict[key].append(x[start_idx:last_idx]) #close up the last one

        return task_dict

    def taskdict2vect(self,task_dict):
        # returns x,y from a dict
        x = []
        y = []
        for y_str, x_arr in task_dict.items():
            for x_row in x_arr:
                x.append(x_row)
                y.append(self.task_str2arr(y_str))
        return np.array(x),np.array(y)
