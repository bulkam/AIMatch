import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
import dataset
import model as m
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()


ds = dataset.Dataset("./Data/AI Match Results 150years_appended_WC2022.txt")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# output path for saving model
model_path = 'models/00/'
model_name = '00.hdf5'

model = m.prepare_model_custom(ds.input_d())

# setup optimizer and compile model
sgd = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
# compile model
model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer=sgd)

print(model.summary())

cpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, model_name),
                                         monitor='loss',
                                         save_best_only=True,
                                         verbose=1)

# train model
callback_history = model.fit(ds, batch_size=10000, epochs=400,
                             callbacks=[cpoint])
# save information about training in txt files
loss_history = callback_history.history["loss"]
val_loss_history = callback_history.history["val_loss"]
numpy_loss_history = np.array(loss_history)
numpy_val_loss_history = np.array(val_loss_history)
np.savetxt(os.path.join(model_path + "/loss_history.txt"), numpy_loss_history, delimiter=",")
np.savetxt(os.path.join(model_path + "/val_loss_history.txt"), numpy_val_loss_history, delimiter=",")