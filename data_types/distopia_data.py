from data_types import Data
from distopia.app.agent import VoronoiAgent
import numpy as np
import pickle as pkl
import csv
from tqdm import tqdm
from multiprocessing import Pool, Manager
from threading import Thread
from environments.distopia_environment import DistopiaEnvironment


class DistopiaData(Data):
    def __init__(self):
        self.voronoi = VoronoiAgent()
        self.voronoi.load_data()
    def set_params(self, specs):
        for param, param_val in specs.items():
            setattr(self, param, param_val)
        self.preprocessors = specs["preprocessors"]
        # if "n_workers" in specs:
        #     self.n_workers = specs["n_workers"]
        # else:
        #     self.n_workers = 1
    def load_data(self, fname, fmt=None, labels_path=None, append=False, load_designs=False, load_metrics=False, load_fiducials=False):
        if fmt is None:
            fmt = self.infer_fmt(fname)
        print(fmt)
        if fmt == "pkl" or fmt== "pk":
            _,raw_data = self.load_pickle(fname)
            for preprocessor in self.preprocessors:
                raw_data = getattr(self,preprocessor)(raw_data) #design_dict2mat_labelled(raw_data)
            self.x,self.y = raw_data
            # try to force de-allocation
            raw_data = None
        elif fmt == "npy":
            assert labels_path is not None
            self.x = np.load(fname)
            self.y = np.load(labels_path)
            for preprocessor in self.preprocessors:
                self.x,self.y = getattr(self,preprocessor)((self.x,self.y))
        elif fmt == "json": #it's a log file (for now)
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
                    if len(log['fiducials']['fiducials']) < 8:
                        continue
                    step_tuple = []
                    if load_designs == True:
                        step_districts = self.jsondistricts2mat(log['districts']['districts'])
                        step_tuple.append(step_districts)
                    if load_metrics == True:
                        assert hasattr(self,"metric_names")
                        step_metrics= DistopiaEnvironment.extract_metrics(self.metric_names,log['districts']['metrics'],
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
            for trajectory in trajectories:
                samples,task = trajectory
                for sstep in samples:
                    self.x.append(*sstep) #any sample data
                    self.y.append(task) #task
            self.x = np.array(self.x)
            self.y = np.array(self.y)

        elif fmt == "csv":
            # TODO: no pre-processing for now, let's fix this later.
            assert labels_path is not None
            if append == False or not hasattr(self,'x') or not hasattr(self,'y'):
                self.x = self.load_csv(fname)
                self.y = self.load_csv(labels_path)
            else:
                self.x = np.concatenate((self.x, self.load_csv(fname)))
                self.y = np.concatenate((self.y, self.load_csv(labels_path)))

    # pre-processing functions
    def save_csv(self,xfname,yfname):
        with open(xfname+".csv", 'w+') as samplefile:
            with open(yfname+"_labels.csv", 'w+') as labelfile:
                samplewriter = csv.writer(samplefile)
                labelwriter = csv.writer(labelfile)
                for i,sample in enumerate(self.x):
                    samplewriter.writerow(sample.flatten())
                    labelwriter.writerow(self.y[i])
    def save_npy(self,xfname,yfname):
        np.save(xfname, self.x)
        np.save(yfname, self.y)
    
    # pre-process labels
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

    def get_task_dict(self, merge=False):
        '''Get a dictionary of trajectories keyed on task
            if merge is True, then concat the trajectories
        '''
        task_dict = dict()
        task_keys = set()
        # start by getting the list of tasks in the data
        for task in self.y:
            task_keys.add(self.task_arr2str(task))

        for key in task_keys:
            task_arr = self.task_str2arr(key)
            indices = np.where((self.y == task_arr).all(axis=1))[0]
            if merge:
                task_dict[key] = self.x[indices]
            # otherwise, we have to split the indices
            else:
                task_dict[key] = []
                last_idx = indices[0]
                start_idx = indices[0]
                for idx in indices[1:]:
                    if idx - last_idx > 1:
                        end_idx = last_idx
                        if end_idx-start_idx > 1:
                            task_dict[key].append(self.x[start_idx:end_idx])
                        start_idx = idx
                    last_idx = idx
                if start_idx < last_idx:
                    task_dict[key].append(self.x[start_idx:last_idx]) #close up the last one
        
        return task_dict