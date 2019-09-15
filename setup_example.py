import os
import sys
import yaml
import shutil
import argparse

example_path = './experiments/example.spec.yaml'
example_log_path = './experiments/logs/example'
template_path = './experiments/examples/example.spec.yaml'

argparser = argparse.ArgumentParser(description="setup or cleanup example experiment spec files")
argparser.add_argument('task', nargs='?', default=None, help="Specify the task. Default (blank) copies example.spec.yaml to experiments root. 'clean' will delete it, if it exists.")
args = argparser.parse_args()
tasks = ['clean']
if args.task is not None and args.task not in tasks:
    raise(ValueError("Invalid task '{}'. Valid tasks are: {}.".format(args.task, tasks)))

if(args.task == 'clean'):
    try:
        os.remove(example_path)
        print("Deleted example file.")
    except FileNotFoundError:
        print("Tried to run clean, but example file does not exist.")

elif os.path.exists(example_path):
    print("Tried to create example file, but example file already exists.")
    sys.exit()

else:
    if os.path.exists(example_log_path):
        # remove the example log dir and its contents
        shutil.rmtree(example_log_path)
    with open(template_path, 'r') as instream:
        spec_dict = yaml.safe_load(instream)
    with open(example_path, 'w') as outstream:
        yaml.safe_dump(spec_dict, outstream)
    print("Removed example log directory and copied template example file to experiments directory.")
