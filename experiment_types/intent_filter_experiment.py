from experiment_types import Experiment
from sklearn.model_selection import train_test_split
from importlib import import_module
class IntentFilterExperiment:
    def run(self, specs):
        # load trajectory data
        data_type = import_module(specs.data_type)
        trajectories = data_type()
        trajectories.load_data(specs.trajectory_file)
        x = trajectories.x
        # split into train/test
        X_train, X_test, y_train, y_test = train_test_split(trajectories.x, trajectories.y, 
                                                test_size=specs.test_size, random_state=specs.random_seed, shuffle=True)
        # record the split somehow (actually if I record the seed I don't need to)
        classifer = specs.model
        classifier.set_params(specs)
        # train model on training data
        classifier.fit(X_train,y_train)
        # test model on test data
        classifier.evaluate(X_test,y_test)

    def log_data(self, data):
        # what is it that I want to log here?
        # sample, prediction, target
        pass
