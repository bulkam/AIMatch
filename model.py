from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ReLU, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Reshape, Resizing
from keras.activations import tanh


def prepare_model_custom(input_d):
    model = Sequential()

    # define model architecture
    model.add(Dense(input_d))
    model.add(Dense(22))
    model.add(ReLU())
    model.add(Dense(2, activation='relu'))

    return model