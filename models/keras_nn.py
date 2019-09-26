from models import Model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D, Conv2D

from copy import deepcopy

class KerasSequential(Model):
    def setup_model(self,specs):
        self.model = Sequential()
        for layer in specs['layers']:
            self.model.add(self.generate_layer(layer))
        self.model.compile(loss=specs['loss'],optimizer=specs['optimizer'])
        self.model.summary()

    def set_params(self,specs):
        #self.specs = deepcopy(specs) # Not used for now, but may do this later. (problem w/ deepcopy and objects in specs, really need to get objects out of specs)
        self.setup_model(specs)
    
    def fit(self,x,Y,fit_params=None):
        if fit_params is None:
            return self.model.fit(x,Y)
        else:
            return self.model.fit(x,Y,**fit_params)

    def save(self,path):
        return self.model.save(path)

    def evaluate(self,x,Y):
        return self.model.evaluate(x,Y)

    def predict(self,x):
        return self.model.predict(x)

    def generate_layer(self,layer):
        layer_type = layer.pop("type")
        if layer_type == "Dense":
            args = ["units", "activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer", "activity_regularizer", "kernel_constraint", "bias_constraint", "input_shape", "batch_input_shape","output_shape"]
            return Dense(**self.extract_layer_args(layer,args,layer_type))
        elif layer_type == "Flatten":
            return Flatten()
        elif layer_type == "Dropout":
            args = ["rate"]
            return Dropout(**self.extract_layer_args(layer,args,layer_type))
        elif layer_type == "Conv2D":
            args = ['filters','kernel_size','strides','padding','data_format','dilation_rate','activation','use_bias','kernel_initializer','bias_initializer','kernel_regularizer','bias_regularizer','activity_regularizer','kernel_constraint','bias_constraint','input_shape','output_shape']
            return Conv2D(**self.extract_layer_args(layer,args,layer_type))

    @staticmethod
    def extract_layer_args(given_args,allowable_args,layer_type):
        extracted_args = {}
        given_args_copy = deepcopy(given_args)
        for key in given_args:
            # TODO: Type-check the args (may need to cast from string anyway. define a dict of types keyed on layer and arg)
            if key in allowable_args:
                extracted_args[key] = given_args_copy.pop(key)
        if len(list(given_args_copy.keys())) > 0:
            print("Warning: layer arguments are defined that {} does not take: {}".format(layer_type,given_args_copy))
        return extracted_args


    