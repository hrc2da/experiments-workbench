import sys
import os
print(sys.path)
print(os.getcwd())
sys.path.append(os.getcwd())
from agents import *
from models.keras_nn import KerasSequential
import yaml




sequential_test_yaml = "tests/test_keras_sequential.yaml"

with open(sequential_test_yaml, 'r') as inputstream:
    all_specs = yaml.safe_load(inputstream)
specs = all_specs['model']
knn = KerasSequential()
knn.set_params(specs)

sequential_test_cnn_yaml = "tests/test_keras_sequential_cnn.yaml"

with open(sequential_test_cnn_yaml, 'r') as inputstream:
    all_specs = yaml.safe_load(inputstream)
specs = all_specs['model']
knn = KerasSequential()
knn.set_params(specs)

sequential_test_cnn_distopia_yaml = "tests/test_keras_sequential_cnn_distopia.yaml"

with open(sequential_test_cnn_distopia_yaml, 'r') as inputstream:
    all_specs = yaml.safe_load(inputstream)
specs = all_specs['model']
knn = KerasSequential()
knn.set_params(specs)