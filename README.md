# experiments-workbench
A framework for running HRCD experiments.
## Dependencies
pyaml

## Getting Started

### Install
`git clone https://github.com/hrc2da/experiments-workbench`

`virtualenv -p python3 <venv name>`

`. <venv name>/bin/activate`

`pip install -r requirements.txt`

### Testing
To define an experiment, create a `*.spec.yaml` file in the experiments directory. See the `experiments/examples` directory for an example.

If your experiment type is not yet defined, you need to write a class extending Experiment in `experiment_types`.

To run all the experiments specified by `.spec.yaml` files in the experiments directory, call `run_experiments.py`.

To run the example, first call `setup_example.py` and then `run_experiments.py`.

All experiment parameters should be defined first in `run_spec.yaml`.

See `__init__.py` in the environments and agents folders for abstract class definitions.

