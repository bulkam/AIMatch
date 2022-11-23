import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
import dataset
import model as m
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import data_preprocessing

X_train, Y_train, X_val, Y_val, X_test = data_preprocessing.Dataset().get_input_data()

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()


tds = dataset.FootballDataset(X_train, Y_train)
vds = dataset.FootballDataset(X_val, Y_val)

print(tds.__getitem__(0)[0].shape, tds.__getitem__(0)[1].shape)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# output path for saving model
model_path = 'models/03/'
model_name = '00.hdf5'

model = m.prepare_model_custom(tds.input_size(), model_path)


# setup optimizer and compile model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)


# compile model
model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer=adam)

print(model.summary())
print(len(tds))
print(len(vds))

cpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, model_name),
                                         monitor='val_loss',
                                         save_best_only=True,
                                         verbose=1)

# train model
callback_history = model.fit(tds, batch_size=1000, epochs=100, validation_data=vds,
                             callbacks=[cpoint])

# save information about training in txt files
loss_history = callback_history.history["loss"]
val_loss_history = callback_history.history["val_loss"]
numpy_loss_history = np.array(loss_history)
numpy_val_loss_history = np.array(val_loss_history)
np.savetxt(os.path.join(model_path + "/loss_history.txt"), numpy_loss_history, delimiter=",")
np.savetxt(os.path.join(model_path + "/val_loss_history.txt"), numpy_val_loss_history, delimiter=",")