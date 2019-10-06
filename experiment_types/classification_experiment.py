from experiment_types import Experiment
from sklearn.model_selection import train_test_split
import pickle as pkl
from utils import import_class
import os
import numpy as np


class DistopiaClassificationExperiment(Experiment):
    def run(self,specs):
        print("Running Classification Experiment")
        # load data
        data_specs = specs['data']
        data_type = data_specs["backend"]
        print(data_type)
        self.data = data_type()
        self.data.set_params(data_specs)
        if "training_labels_path" in data_specs:
            labels_path = data_specs["training_labels_path"]
        else: 
            labels_path = None
        self.data.load_data(data_specs["training_path"],labels_path=labels_path)
        # split the data
        if "test_path" in data_specs:
            x_train = self.data.x
            y_train = self.data.y
            self.test_data = data_type()
            self.test_data.set_params(data_specs)
            self.test_data.preprocessors = data_specs["test_preprocessors"] #hackity hack hack hack
            if "test_labels_path" in data_specs:
                test_labels_path = data_specs["test_labels_path"]
            else:
                test_labels_path = None
            self.test_data.load_data(data_specs["test_path"], labels_path=test_labels_path)
            x_test = self.test_data.x
            y_test = self.test_data.y
        else:
            x_train, x_test, y_train, y_test = train_test_split(self.data.x, self.data.y, 
                                                test_size=data_specs["test_proportion"], random_state=specs['random_seed'], shuffle=True)
        # initialize model
        model_specs = specs['model']
        model_type = model_specs["backend"]
        model = model_type()
        model.set_params(model_specs)
        # train the model
        #n,width,height = x_train.shape
        #tn,twidth,theight = x_test.shape
        fit_params = model_specs["fit_params"]
        fit_params["validation_data"] = (x_test, y_test)
        #history = model.fit(x_train.reshape(n,width,height,1),y_train, fit_params)
        history = model.fit(x_train,y_train, fit_params)
        with open(os.path.join(specs['logpath'],'history.pkl'), 'wb') as outfile:
            pkl.dump(history.history,outfile)
        model.save(os.path.join(specs['logpath'],'model.h5'))
        # test the model
        #n,width,height = x_test.shape
        test_mse = model.evaluate(x_test,y_test)
        result_str = "Test MSE: {}".format(test_mse)
        print(result_str)
        with open(os.path.join(specs['logpath'],'test_results'),'w+') as outfile:
            outfile.write(result_str)
        predictions = model.predict(x_test)
        np.save(os.path.join(specs['logpath'],'test_predictions'),predictions)
        np.save(os.path.join(specs['logpath'],'test_labels'),y_test)        


