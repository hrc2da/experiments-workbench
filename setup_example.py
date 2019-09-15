import os
import sys
import yaml
import shutil

example_path = './experiments/example.spec.yaml'
example_log_path = './experiments/logs/example'
template_path = './experiments/examples/example.spec.yaml'
if os.path.exists(example_path):
    sys.exit()

else:
    if os.path.exists(example_log_path):
        # remove the example log dir and its contents
        shutil.rmtree(example_log_path)
    with open(template_path, 'r') as instream:
        spec_dict = yaml.safe_load(instream)
    with open(example_path, 'w') as outstream:
        yaml.safe_dump(spec_dict, outstream)
