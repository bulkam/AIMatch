from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ReLU, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, Resizing
from keras.activations import tanh


def prepare_model_custom(input_shape, model_path):
    if model_path == "models/00/":
        model = Sequential()
        # define model architecture
        model.add(Input(input_shape))
        model.add(Dense(input_shape[0]))
        model.add(Dense(22))
        model.add(Dense(2))

        return model

    if model_path == "models/01/":
        model = Sequential()
        # define model architecture
        model.add(Input(input_shape))
        model.add(Dense(input_shape[0]))
        model.add(Dense(2))

        return model

    if model_path == "models/02/":
        model = Sequential()
        # define model architecture
        model.add(Input(input_shape))
        model.add(Dense(input_shape[0]))
        model.add(Dense(128))
        model.add(ReLU())
        model.add(Dense(22))
        model.add(Dense(2))

        return model

    if model_path == "models/03/":
        model = Sequential()
        # define model architecture
        model.add(Input(input_shape))
        model.add(Dense(input_shape[0]))
        model.add(Dense(1024))
        model.add(ReLU())
        model.add(Dense(16))
        model.add(Dense(2))

        return model
