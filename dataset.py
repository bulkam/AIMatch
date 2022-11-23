import numpy as np
import tensorflow as tf


class FootballDataset(tf.keras.utils.Sequence):
    def __init__(self, x, y, prediction=False, batch_size=1):
        super().__init__()
        self.x = x
        if not prediction:
            self.y = y
        else:
            self.y = np.array([[0, 0] for i in range(len(x))])

    def input_size(self):
        return self.x[0].shape

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array([self.x[idx]]), np.array([self.y[idx]])
