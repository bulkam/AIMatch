import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

ds = dataset.Dataset("./Data/AI Match Results 150years_appended_WC2022.txt")

model_path = "models/00"
model_name = "00.hdf5"

home_team_name = "Qatar"
away_team_name = "Ecuador"

home_team = []
away_team = []

for i in range(ds.team_count):
    home_team.append(0)
    away_team.append(0)
    if i == ds.team_numbers[home_team_name]:
        print(i)
        home_team[i] = 1
    if i == ds.team_numbers[away_team_name]:
        print(i)
        away_team[i] = 1

x = np.array([np.append(home_team, away_team)])

model = keras.models.load_model(os.path.join(model_path, model_name))
model.build(x.shape)

y = model.predict(x)
print(y)
