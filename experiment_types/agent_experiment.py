from experiment_types import Experiment
import datetime
import os
import json
# from multiprocessing import Pool, Manager
# from threading import Thread
#from tqdm import tqdm

def run_agent(specs,task):
    # runs an agent with a set task
    log_data = {}  # init log data
    log_data['experiment_info'] = specs['experiment_description']
    log_data['start_time'] = datetime.datetime.now()
    log_data['metric_names'] = specs['environment_params']['metrics']
    log_data['task'] = task
    log_data['episodes'] = []
    # run the agent
    agent = specs['agent']()
    environment = specs['environment']()
    environment.set_params(specs['environment_params'])

    agent.set_params(specs['agent_params'])
    agent.set_task(task)
    if 'random_seed' in specs:
        environment.seed(specs['random_seed'])
        agent.seed(specs['random_seed'])
    design, metric, reward, episodes = agent.run(environment, specs)
    this_ep={}
    ep_counter=1
    this_ep['episode_no'] = ep_counter
    this_ep['run_log'] = []

    for ind, m in enumerate(metric):
        this_step={}
        this_step['step_no'] = ind
        this_step['metrics'] = m
        this_step['design'] = design[ind]
        this_step['reward'] = reward[ind]
        this_ep['run_log'].append(this_step)
        if (ind+1) % specs['agent_params']['episode_length'] == 0:
            log_data['episodes'].append(this_ep)
            ep_counter = ep_counter+1
            this_ep={}
            this_ep['episode_no']=ep_counter
            this_ep['run_log'] = []
    log_data['end_time'] = datetime.datetime.now()
    return log_data


class AgentExperiment(Experiment):

    def run(self, specs):
        # set the logpaths (we don't use these anyway...we should, it would be cleaner)
        specs['environment_params']['logfile'] = specs['logpath']
        specs['agent_params']['logfile'] = specs['logpath']
        print('='*30)
        print(specs)
        if 'n_workers' in specs:
            n_workers = specs['n_workers']
            print("ALLOCATING N_WORKERS!!: {}".format(n_workers))
        else:
            print("ONLY ONE WORKER!!")
            n_workers = 1
        # helper function to run the experiments in parallel

        if 'tasks' in specs['agent_params']:
            tasks = specs['agent_params']['tasks']
        else:
            tasks = [specs['agent_params']['task']]
        n_samples = len(tasks)*specs['agent_params']['num_episodes']*specs['agent_params']['episode_length']
#        progress_queue = Manager().Queue()
        # progress_thread = Thread(target=self.progress_monitor,args=(n_samples,progress_queue))
        # progress_thread.start()
        # agent_runners = []
        # for task in tasks:
        #     agent_runners.append((specs,task,progress_queue))
        #
        # with Pool(n_workers) as pool:
        #     results = pool.starmap(run_agent, agent_runners)
        results = [run_agent(specs,tasks[0])]
        curr_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        file_name = curr_time + '.json'
        logpath = specs['logpath']
        with open(os.path.join(logpath, file_name), 'w') as outfile:
            json.dump(results, outfile, default=str, indent=4)

    # def progress_monitor(self,n_samples,progress_queue):
    #     for i in tqdm(range(n_samples)):
    #         progress_queue.get()
