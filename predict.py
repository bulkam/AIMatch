import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data_preprocessing
import keras
import dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()


preprocessed_data = data_preprocessing.Dataset()
X_train, Y_train, X_val, Y_val, X_test = preprocessed_data.get_input_data()
all_teams = preprocessed_data.all_teams

ds = dataset.FootballDataset(X_test, None, prediction=True)

model_path = "models/03/"
model_name = "00.hdf5"

model = keras.models.load_model(os.path.join(model_path, model_name))
model.build(ds.input_size())

for x, y in ds:
    home_team_part = x[0][:len(all_teams)]
    away_team_part = x[0][len(all_teams):2 * len(all_teams)]
    home_team = all_teams[np.where(home_team_part == 1)[0][0]]
    away_team = all_teams[np.where(away_team_part == 1)[0][0]]
    y = model.predict(x)
    print(home_team, " x ", away_team, ": ", y)
    with open(os.path.join("./predictions/", model_path.split('/')[1] + ".txt"), "a") as f:
        f.write(home_team + " x " + away_team + ": " + str(y))
