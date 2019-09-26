class Model:
    def set_params(self,specs):
        raise(NotImplementedError)
    def setup_model(self):
        raise(NotImplementedError)
    def fit(self):
        raise(NotImplementedError)

# add any models you write to this list and __all__
import models.keras_nn
__all__ = ['keras_nn']
