from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam


optimizer = Adam(lr=0.001)
model = Sequential()
model.add(Dense(64, input_dim=40))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer=optimizer)
print("PLOTTING")
from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='NN_model.png')
print("PLOTTED")