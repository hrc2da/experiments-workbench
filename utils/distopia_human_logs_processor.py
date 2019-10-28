from glob import glob
import sys
import os
#sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
sys.path.append(os.path.normpath(os.getcwd()))

from data_types.distopia_data import DistopiaData


# filename convention is attribute_description_end.json

#attribute is fixed (only one word), followed by either some human name or "agent"

#description is variable length, each word seperated by _, for other unique info you want to put
#Ex. maybe you want to put the constants you used for your SARSA agent

#end is fixed length (one word) this be whatever you want: maybe the task that you/your agent was trying to solve,
#if the log only was for one task. Maybe it's the task index of the last task finished in this log
#End is not really relevant to this file, just for you to put more info, but please put something

#The filename is attribute + description. If two log files have the same name + description,
#the script will consider both logs to be part of the same process and will combine them
#into the same CSV files (whose names will be attribute + description + 'logs.csv')

#Ex. agent_sarsa_1000_0.2_0.9_[0.0.1].json --> "agent" is attribute, "sarsa_1000_0.2_0.9" is description, "[0.0.1]" is End
#agent_sarsa_1000_0.2_0.9_[0.0.1].json and agent_sarsa_1000_0.2_0.9_[1.0.1].json
#will be treated as two seperate logs for the same experiment and will combine them when making CSVs
#agent_sarsa_1000_0.2_0.9_[0.0.1].json and agent_sarsa_1000_0.9_0.2_[0.0.1].json will be treated as two seperate experiments.

def logs_processor(data_dir, norm_dir):
    is_human = 0
    logpaths = sorted(glob(os.path.join(data_dir,'*.json')))
    print(logpaths)
    data = DistopiaData()
    cur_metrics = ['population', 'pvi', 'compactness']
    data.set_params({'metric_names': cur_metrics,'preprocessors':[]})
    for i in range(len(logpaths)):
        logfile = logpaths[i]
        print(logfile)
        new_user=False
        if os.name == "nt":
            cur_file = logfile.split('\\')[-1]
        else:
            cur_file = logfile.split('/')[-1]
        split_file = cur_file.split('_')
        file_attr = split_file[0]
        file_ext =  split_file[-1]
        file_desc = '_'.join(split_file[1:-1])
        filename = file_attr + "_" + file_desc
        print("Filename:" + filename)
        if file_attr == "agent":
            is_human=0
        else:
            is_human=1
        if i < len(logpaths)-1:
            if os.name == "nt":
                next_file = logpaths[i+1].split('\\')[-1]
            else:
                next_file = logpaths[i+1].split('/')[-1]
            next_split_file = next_file.split('_')
            next_file_attr = next_split_file[0]
            next_file_ext = next_split_file[-1]
            next_file_desc = '_'.join(next_split_file[1:-1])
            next_filename = next_file_attr + "_" + next_file_desc
        if i==len(logpaths)-1 or filename != next_filename:
            new_user=True
        if is_human:
            data.load_data(logfile,append=True,load_designs=False,load_metrics=True, norm_file = norm_dir)
        else:
            data.load_agent_data(logfile,append=True,load_designs=False,load_metrics=True, norm_file = norm_dir)
        if new_user == True:
            fname = os.path.join(data_dir,"{}_logs".format(filename))
            print(fname)
            data.save_csv(fname,fname)
            del data
            data= DistopiaData()
            data.set_params({'metric_names': cur_metrics,'preprocessors':[]})
